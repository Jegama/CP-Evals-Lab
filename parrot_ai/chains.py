"""Multi-step reasoning chains for enhanced theological responses (language-aware).

Each ParrotAI / ParrotAIHF instance carries its own prompts module (see core._load_prompts).
This file reads those dynamically so adding a new language only requires a new prompts/<lang>.py.
"""

from typing import Union, Any
from .core import LocalModelParrotAI, ParrotAIHF, ParrotAIOpenAI, ParrotAITogether, ParrotAIGemini, ParrotAIGrok  # lightweight (ParrotAI heavy deps are lazy)


ProviderType = Union[LocalModelParrotAI, ParrotAIHF, ParrotAIOpenAI, ParrotAITogether, ParrotAIGemini, ParrotAIGrok]


def _prompts(parrot_instance: ProviderType):
    mod = getattr(parrot_instance, "prompts", None)
    if mod is None:
        raise ValueError("Parrot instance has no 'prompts' module loaded.")
    return mod

def _require(mod: Any, name: str):
    if not hasattr(mod, name):
        raise ValueError(f"Prompts module missing required attribute: {name}")
    return getattr(mod, name)


def parrot_chain(data, parrot_instance: ProviderType):
    """
    Execute a multi-step reasoning chain for theological questions.
    
    This chain generates multiple candidate answers, reviews them through
    a Calvin-based theological lens, and produces a final refined answer.
    
    Args:
        data: List containing the conversation data, where data[0] is the user question
              and data[1] is the verified answer from the dataset
        parrot_instance: An initialized ParrotAI instance with a loaded model
    
    Returns:
        dict: Contains all intermediate outputs and the final answer
    """
    if not parrot_instance.is_loaded():
        raise ValueError("ParrotAI instance must have a loaded model")
    
    question = data[0]["content"]
    P = _prompts(parrot_instance)

    reasoning_tpl = _require(P, "reasoning_prompt")
    calvin_sys = _require(P, "CALVIN_SYS_PROMPT")
    main_sys = _require(P, "MAIN_SYSTEM_PROMPT")
    review_tpl = _require(P, "calvin_review_prompt")
    final_tpl = _require(P, "final_answer_prompt")

    reasoning_prompt = reasoning_tpl.format(user_question=question)

    # Verified answer from the dataset
    first_answer = data[1]["content"]

    # Second answer using the main system prompt
    second_answer = parrot_instance.generate(reasoning_prompt, system=main_sys)

    # Third answer using the Calvin system prompt
    third_answer = parrot_instance.generate(reasoning_prompt, system=calvin_sys)

    # Step 2: Calvin Review (depends on all three answers)
    review_prompt = review_tpl.format(
        user_question=question,
        first_answer=first_answer,
        second_answer=second_answer,
        third_answer=third_answer,
    )
    calvin_review = parrot_instance.generate(review_prompt, system=calvin_sys)

    # Step 3: Final Answer (depends on the review)
    final_answer_prompt = final_tpl.format(
        user_question=question,
        first_answer=first_answer,
        second_answer=second_answer,
        third_answer=third_answer,
        calvin_review=calvin_review,
    )

    final_answer = parrot_instance.generate(final_answer_prompt, system=main_sys)

    return {
        "first_answer": first_answer,
        "second_answer": second_answer,
        "third_answer": third_answer,
        "calvin_review": calvin_review,
        "final_answer": final_answer
    }


def simple_chain(question: str, parrot_instance: ProviderType):
    """
    Execute a simple single-step generation for quick responses.
    
    Args:
        question: The user's question as a string
        parrot_instance: An initialized ParrotAI instance with a loaded model
    
    Returns:
        str: The generated response
    """
    if not parrot_instance.is_loaded():
        raise ValueError("ParrotAI instance must have a loaded model")
    
    P = _prompts(parrot_instance)
    reasoning_tpl = _require(P, "reasoning_prompt")
    main_sys = _require(P, "MAIN_SYSTEM_PROMPT")
    reasoning_prompt = reasoning_tpl.format(user_question=question)
    
    return parrot_instance.generate(reasoning_prompt, system=main_sys)


def comparative_chain(question: str, parrot_instance: ProviderType, system_prompts: list):
    """
    Generate responses using multiple system prompts for comparison.
    
    Args:
        question: The user's question
        parrot_instance: An initialized ParrotAI instance with a loaded model
        system_prompts: List of system prompts to use for generation
    
    Returns:
        dict: Mapping of prompt names to generated responses
    """
    if not parrot_instance.is_loaded():
        raise ValueError("ParrotAI instance must have a loaded model")
    
    P = _prompts(parrot_instance)
    reasoning_tpl = _require(P, "reasoning_prompt")
    reasoning_prompt = reasoning_tpl.format(user_question=question)
    
    results = {}
    for i, system_prompt in enumerate(system_prompts):
        response = parrot_instance.generate(
            reasoning_prompt,
            system=system_prompt
        )
        results[f"response_{i+1}"] = response
    
    return results
