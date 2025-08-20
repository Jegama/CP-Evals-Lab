"""ParrotAI package public API.

We keep imports light so that users performing only evaluation / HF API usage
don't need heavy local model deps (torch, transformers, bitsandbytes) installed.

``ParrotAI`` (local) still available via standard import; its heavy deps are
now lazily imported inside ``ParrotAI.load_model`` (see ``core.py``).
"""

from .core import LocalModelParrotAI, ParrotAIHF, ParrotAIOpenAI, ParrotAITogether, ParrotAIGemini, ParrotAIGrok  # noqa: F401
from .chains import parrot_chain  # noqa: F401

# Backward compatibility alias
ParrotAI = LocalModelParrotAI

__version__ = "0.1.1"
__all__ = ["ParrotAI", "LocalModelParrotAI", "ParrotAIHF", "ParrotAIOpenAI", "ParrotAITogether", "ParrotAIGemini", "ParrotAIGrok", "parrot_chain"]
