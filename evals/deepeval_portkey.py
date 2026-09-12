
"""
DeepEval evaluation LLM using Portkey -> Groq.

Portkey integration:
    @aws-rag-app

Underlying model:
    openai/gpt-oss-20b

The wrapper implements DeepEval's schema-aware interface so that
metrics such as Faithfulness can receive valid Pydantic objects
instead of trying to parse arbitrary model text.
"""

import json
import os
from typing import Optional, Type

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from deepeval.models.base_model import DeepEvalBaseLLM


load_dotenv()


PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")

if not PORTKEY_API_KEY:
    raise RuntimeError(
        "PORTKEY_API_KEY is not set. "
        "Make sure it exists in your .env file."
    )


PORTKEY_BASE_URL = "https://api.portkey.ai/v1"

PORTKEY_PROVIDER = "@aws-rag-app"

EVALUATION_MODEL = "openai/gpt-oss-20b"


def _extract_json_data(content: str) -> dict:
    content_str = content.strip()
    try:
        return json.loads(content_str)
    except json.JSONDecodeError:
        pass

    import re
    match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", content_str, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"(\{.*\}|\[.*\])", content_str, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Evaluation model returned invalid JSON.\n\n"
        f"Raw output:\n{content}"
    )


class PortkeyGroqEvaluationModel(DeepEvalBaseLLM):
    """
    DeepEval LLM wrapper.

    Portkey routes the request to the Groq integration configured
    under @aws-rag-app.
    """

    def __init__(
        self,
        model: str = EVALUATION_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=PORTKEY_API_KEY,
            base_url=PORTKEY_BASE_URL,
            default_headers={
                "x-portkey-provider": PORTKEY_PROVIDER,
            },
        )

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ):
        """
        Generate a response.

        If DeepEval supplies a Pydantic schema, request JSON and
        convert the response into the supplied schema.

        Otherwise return plain text.
        """

        messages = []
        if schema is not None:
            messages.append({
                "role": "system",
                "content": (
                    "You are an evaluation AI assistant. "
                    "You MUST respond strictly with valid JSON format."
                ),
            })
            prompt_content = prompt
            if "json" not in prompt.lower():
                prompt_content = f"{prompt}\n\nIMPORTANT: Respond strictly in valid JSON format."
            messages.append({"role": "user", "content": prompt_content})
        else:
            messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # DeepEval metrics such as Faithfulness provide a schema.
        # Ask the model for JSON in those cases.
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_object"
            }

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            # If Groq fails JSON mode validation (e.g., json_validate_failed 400 error),
            # retry without response_format and rely on python JSON extractor.
            if schema is not None and "response_format" in kwargs:
                kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise exc

        content = response.choices[0].message.content

        if content is None:
            raise ValueError("Evaluation model returned empty content.")

        # Normal text generation.
        if schema is None:
            return content

        # Structured generation.
        data = _extract_json_data(content)

        try:
            return schema.model_validate(data)
        except Exception:
            # Compatibility with older Pydantic versions.
            return schema.parse_obj(data)

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ):
        """
        DeepEval's async interface.

        We intentionally call the synchronous implementation.

        The evaluation script sets async_mode=False to avoid
        concurrent Groq requests and TPM exhaustion.
        """

        return self.generate(prompt, schema)

    def get_model_info(self):
        return {
            "provider": "Portkey",
            "integration": PORTKEY_PROVIDER,
            "model": self.model_name,
        }


evaluation_model = PortkeyGroqEvaluationModel()


def print_model_info():
    info = evaluation_model.get_model_info()

    print("=" * 80)
    print("DeepEval Evaluation LLM")
    print("=" * 80)
    print(f"Provider    : {info['provider']}")
    print(f"Integration : {info['integration']}")
    print(f"Model       : {info['model']}")
    print("=" * 80)

