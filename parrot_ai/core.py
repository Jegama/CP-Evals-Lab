"""Core runtime classes for ParrotAI.

Design goal: Avoid importing heavy local model dependencies (torch, transformers,
bitsandbytes) unless the user explicitly chooses to load a local model via
``ParrotAI.load_model``. This lets light‑weight usages (e.g. evaluation with
OpenAI / Together / HF API) work in environments where ``torch`` is not
installed.

ParrotAI (local model):
  - Heavy deps are imported lazily inside ``load_model``.
  - If a user calls any generation API before ``load_model`` an error is raised.

ParrotAIHF (HF Inference API):
    - Only depends on ``huggingface_hub`` (lightweight) and can be used without
        ``torch`` installed.

ParrotAIOpenAI / ParrotAITogether:
    - Lightweight API wrappers mirroring ParrotAIHF interface so higher-level
        chains / dataset creators can swap providers uniformly.
    - Environment variables required:
                OPENAI_API_KEY  (for OpenAI)
                TOGETHER_API_KEY (for Together AI)
        These are loaded via python-dotenv if present.
"""

import os
import importlib
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from typing import Any, cast, Optional
from contextlib import suppress

# --- Prompt module loader -------------------------------------------------
def _load_prompts(language: str):
    """Import the prompt module for a given language or raise a clear error.

    Required symbols the module should define:
      - MAIN_SYSTEM_PROMPT
      - CALVIN_SYS_PROMPT
      - reasoning_prompt
      - calvin_review_prompt
      - final_answer_prompt
    """
    module_name = f"parrot_ai.prompts.{language}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ImportError(
            f"Prompt module '{module_name}' not found. Create 'parrot_ai/prompts/{language}.py' "
            "with required constants: MAIN_SYSTEM_PROMPT, CALVIN_SYS_PROMPT, reasoning_prompt, "
            "calvin_review_prompt, final_answer_prompt."
        ) from e


class LocalModelParrotAI:
    """Local model wrapper with (optional) 4-bit quantization support.

    The class defers importing heavy libraries until ``load_model`` is called to
    keep ``import parrot_ai`` light for users who only need API-backed flows.
    """
    def __init__(self, language: str = "arabic"):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self._torch = None  # will be set after lazy import in load_model
        self.language = language
        self.prompts = _load_prompts(language)

    def load_model(self, model_name: str):
        """Load a causal LM with 4-bit quantization (requires torch + transformers).

        Imports torch/transformers/bitsandbytes lazily so the package can be
        imported without those heavy dependencies present.
        """
        try:  # Lazy heavy imports
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            from transformers.utils.quantization_config import BitsAndBytesConfig  # type: ignore
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "Local model loading requires 'torch' and 'transformers'. Install them, e.g.\n"
                "  pip install torch transformers accelerate bitsandbytes\n"
                "(choose the correct torch build for your platform/GPU)."
            ) from e

        self._torch = torch

        # Clear GPU cache if available (safe no-op on CPU‑only builds)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name
        print(f"Model {model_name} loaded successfully!")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ):
        """One-shot text generation, chat-template aware.

        Returns only the assistant reply text.
        """
        if self.model is None or self.tokenizer is None or self._torch is None:
            raise ValueError("Model not loaded. Call load_model() first (requires torch).")

        # Resolve system prompt (explicit > default from prompts module > none)
        system_prompt = system if system is not None else getattr(self.prompts, "MAIN_SYSTEM_PROMPT", "")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([chat], return_tensors="pt").to(self.model.device)

        torch = self._torch  # local alias
        with torch.no_grad():  # type: ignore[attr-defined]
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        reply_ids = gen_ids[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def get_model_info(self) -> str:
        if not self.is_loaded():
            return "No model loaded"
        assert self.model is not None
        info_lines = [
            f"Model Name: {self.model_name}",
        ]
        try:
            info_lines.append(f"Memory Footprint: {self.model.get_memory_footprint() / 1e9:.2f} GB")
            info_lines.append(f"Total Parameters: {self.model.num_parameters():,}")
            info_lines.append(f"Trainable Parameters: {self.model.num_parameters(only_trainable=True):,}")
        except Exception:  # noqa: BLE001
            pass
        cfg = getattr(self.model, 'config', None)
        for attr in [
            'model_type','hidden_size','num_hidden_layers','num_attention_heads',
            'vocab_size','max_position_embeddings'
        ]:
            if cfg is not None and hasattr(cfg, attr):
                info_lines.append(f"{attr.replace('_',' ').title()}: {getattr(cfg, attr)}")
        try:
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            info_lines.extend([f"Device: {device}", f"Data Type: {dtype}"])
        except Exception:  # noqa: BLE001
            pass
        if getattr(self.model, 'is_quantized', False):
            info_lines.append("Quantization: 4-bit (BitsAndBytes)")
        if getattr(self.model, 'can_generate', lambda: False)():
            info_lines.append("Generation: Supported")
        return "\n".join(info_lines)

class BaseParrotAI:
    """Base class for API-based ParrotAI implementations with shared functionality."""
    
    def __init__(self, language: str = "arabic"):
        """Initialize base ParrotAI with common attributes."""
        load_dotenv()
        self.model_name: Optional[str] = None
        self.language = language
        self.prompts = _load_prompts(language)
        self._client = None
    
    def set_model(self, model_name: str):
        """Set the model to use for generation."""
        self.model_name = model_name
        print(f"Model set to: {model_name}")
    
    def _build_messages(self, prompt: str, system: Optional[str] = None):
        """Build messages array for API calls."""
        system_prompt = system if system is not None else getattr(self.prompts, "MAIN_SYSTEM_PROMPT", "")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
    
    def is_loaded(self) -> bool:
        """Check if the API client is initialized."""
        return self._client is not None
    
    def generate(self, prompt: str, system: Optional[str] = None, model: Optional[str] = None, **kwargs) -> str:
        """Generate text using the API. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate method")
    
    def get_model_info(self) -> str:
        """Get information about the current configuration. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_model_info method")


class ParrotAIHF(BaseParrotAI):
    """HuggingFace API wrapper for text generation."""
    
    def __init__(self, provider: str = "nebius", language: str = "arabic"):
        """Initialize ParrotAIHF instance with HuggingFace API client."""
        super().__init__(language)
        
        # Get HF token from environment
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN must be set in environment variables")
        
        self._client = InferenceClient(
            api_key=hf_token,
            provider=cast(Any, provider)
        )
        self.provider = provider
        print("HuggingFace API client initialized")
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text using HuggingFace API."""
        model_to_use = model or self.model_name or "google/gemma-3-27b-it"
        messages = self._build_messages(prompt, system)

        completion = self._client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        return completion.choices[0].message.content or ""
    
    def get_model_info(self) -> str:
        """Get information about the current configuration."""
        if not self.is_loaded():
            return "API client not initialized"
        
        return "\n".join([
            f"Provider: {self.provider}",
            f"Current Model: {self.model_name or 'Not set (will use default)'}",
            "Type: HuggingFace API Client",
        ])


class ParrotAIOpenAI(BaseParrotAI):
    """OpenAI API wrapper for text generation."""

    def __init__(self, language: str = "arabic"):
        super().__init__(language)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("openai package not installed. Add it to requirements.txt") from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY must be set in environment variables")
        self._client = OpenAI()
        print("OpenAI client initialized")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        model_to_use = model or self.model_name or "gpt-5-mini"
        messages = self._build_messages(prompt, system)
        
        resp = self._client.responses.create(
            model=model_to_use,
            input=messages
        )
        answer = getattr(resp, 'output_text', "")
        
        return answer

    def get_model_info(self) -> str:
        return "\n".join([
            "Provider: openai",
            f"Current Model: {self.model_name or 'Not set (will use default)'}",
            "Type: OpenAI Chat Completions Client",
        ])


class ParrotAITogether(BaseParrotAI):
    """Together AI API wrapper for text generation."""

    def __init__(self, language: str = "arabic"):
        super().__init__(language)
        try:
            from together import Together  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("together package not installed. Add it to requirements.txt") from e
        if not os.environ.get("TOGETHER_API_KEY"):
            raise ValueError("TOGETHER_API_KEY must be set in environment variables")
        self._client = Together()
        print("Together AI client initialized")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        model_to_use = model or self.model_name or "google/gemma-3-12b-it"
        messages = self._build_messages(prompt, system)
        
        completion = self._client.chat.completions.create(
            model=model_to_use,
            messages=messages,
        )
        with suppress(Exception):
            # Together returns similar structure with choices
            choices = getattr(completion, 'choices', None)
            if choices:
                msg = getattr(choices[0], 'message', None)
                if msg:
                    content = getattr(msg, 'content', '')
                    if isinstance(content, str):
                        return content
        return ""

    def get_model_info(self) -> str:
        return "\n".join([
            "Provider: together",
            f"Current Model: {self.model_name or 'Not set (will use default)'}",
            "Type: Together AI Chat Completions Client",
        ])


class ParrotAIGemini(BaseParrotAI):
    """Google Gemini API wrapper for text generation."""

    def __init__(self, language: str = "arabic"):
        super().__init__(language)
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("google-genai package not installed. Add it to requirements.txt") from e
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY must be set in environment variables")
        
        self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self._types = types
        print("Gemini client initialized")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        model_to_use = model or self.model_name or "gemini-2.5-flash"
        system_prompt = system if system is not None else getattr(self.prompts, "MAIN_SYSTEM_PROMPT", "")
        
        config = self._types.GenerateContentConfig()
        if system_prompt:
            config.system_instruction = system_prompt
            
        response = self._client.models.generate_content(
            model=model_to_use,
            contents=prompt,
            config=config
        )
        
        return response.text or ""

    def get_model_info(self) -> str:
        return "\n".join([
            "Provider: gemini",
            f"Current Model: {self.model_name or 'Not set (will use default)'}",
            "Type: Google Gemini API Client",
        ])


class ParrotAIGrok(BaseParrotAI):
    """xAI Grok API wrapper for text generation."""

    def __init__(self, language: str = "arabic"):
        super().__init__(language)
        try:
            from xai_sdk import Client  # type: ignore
            from xai_sdk.chat import user, system  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("xai-sdk package not installed. Add it to requirements.txt") from e
        if not os.environ.get("XAI_API_KEY"):
            raise ValueError("XAI_API_KEY must be set in environment variables")
        
        self._client = Client(
            api_key=os.environ.get("XAI_API_KEY"),
            timeout=3600,
        )
        self._user = user
        self._system = system
        print("Grok client initialized")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        model_to_use = model or self.model_name or "grok-3-mini"
        system_prompt = system if system is not None else getattr(self.prompts, "MAIN_SYSTEM_PROMPT", "")
        
        chat = self._client.chat.create(model=model_to_use)
        
        if system_prompt:
            chat.append(self._system(system_prompt))
        chat.append(self._user(prompt))
        
        response = chat.sample()
        return response.content

    def get_model_info(self) -> str:
        return "\n".join([
            "Provider: grok",
            f"Current Model: {self.model_name or 'Not set (will use default)'}",
            "Type: xAI Grok API Client",
        ])