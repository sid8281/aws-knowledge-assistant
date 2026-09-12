from portkey_ai import Portkey
from app.config import settings


client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    virtual_key=settings.PORTKEY_VIRTUAL_KEY_FALLBACK,
)


def generate_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model=settings.LLM_MODEL_FALLBACK,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content