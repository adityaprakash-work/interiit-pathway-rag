# ---INFO-----------------------------------------------------------------------
"""
Programs, guidance or otherwise.
"""


# ---DEPEDENCIES----------------------------------------------------------------
import openai
import guidance

from guidance import (
    gen,
    select,
    system,
    assistant,
    user,
)
from . import settings
from .prompts import hyde_sys_prompt, hyde_proceed_prompt


# ------------------------------------------------------------------------------
openai_settings = settings.OpenAISettings()


# ---HyDE-----------------------------------------------------------------------
@guidance
def hyde_g(
    lm,
    query,
    addendum=None,
    max_tokens=50,
    temperature=0.1,
    is_chat=True,
):
    """
    Program to generate a proxy answer for a query, supporting both chat and
    completion-based models.

    Args:
    lm (guidance.models.Model): The language model.
    query (str): The query.
    addendum (str): Extra considerations.
    max_tokens (int): The maximum number of tokens.
    temperature (float): Sampling temperature.
    is_chat (bool): Whether the model is chat-based.

    Returns:
    lm_hyde (guidance.models.Model): Model with the generated proxy answer.
    """
    sys_prompt = hyde_sys_prompt.format(max_words=int(max_tokens * 0.75))
    proceed_prompt = hyde_proceed_prompt.format(query=query)
    if addendum:
        sys_prompt += f"\n\nExtra concerns: {addendum}"
    if is_chat:
        with system():
            lm_hyde = lm + sys_prompt
        with user():
            lm_hyde += proceed_prompt
        with assistant():
            try:
                lm_hyde += select(["YES", "NO"], name="hyde")
            except:
                lm_hyde += gen("hyde", max_tokens=2)
        if lm_hyde["hyde"] == "NO":
            return lm_hyde
        with assistant():
            lm_hyde += gen(
                "hyde",
                max_tokens=max_tokens,
                temperature=temperature,
            )
    else:
        completion_prompt = f"{sys_prompt}\n\nQuery: {query}\n"
        lm_hyde = lm + completion_prompt
        if lm_hyde["hyde"] == "NO":
            return lm_hyde
        lm_hyde += gen("hyde", max_tokens=max_tokens, temperature=temperature)

    return lm_hyde


def hyde_openai_chat(
    query,
    addendum=None,
    model="gpt-4o-mini",
    max_tokens=50,
    temperature=0.1,
):
    """
    Program to generate a proxy answer for a query using OpenAI's chat-based
    models directly. Upto 8x faster than guidance based hyde.
    """
    sys_prompt = hyde_sys_prompt.format(max_words=int(max_tokens * 0.75))
    proceed_prompt = hyde_proceed_prompt.format(query=query)
    if addendum:
        sys_prompt += f"\n\nExtra concerns: {addendum}"
    client = openai.OpenAI(api_key=openai_settings.api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": proceed_prompt},
        ],
    )
    if "NO" in response.choices[0].message.content:
        return "NO"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"\n\nQuery: {query}\n"},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content
