"""LLM answer generation. Provider is chosen with the LLM_PROVIDER env var."""
import os

SYSTEM_PROMPT = (
    "You answer questions using only the context passages provided. "
    "If the answer is not in the context, say you could not find it in the documents. "
    "Cite sources as [filename p.N] after the sentences they support. Be concise."
)

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
}


def build_context(chunks) -> str:
    return "\n\n".join(
        f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks
    )


def generate_answer(question: str, chunks) -> str:
    provider = os.getenv("LLM_PROVIDER", "none").lower()

    if provider == "none" or not chunks:
        if not chunks:
            return "No documents have been uploaded yet."
        return "LLM is not configured, so here are the most relevant passages found."

    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider)
    user_msg = f"Context:\n{build_context(chunks)}\n\nQuestion: {question}"

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        resp = client.messages.create(
            model=model,
            max_tokens=700,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
