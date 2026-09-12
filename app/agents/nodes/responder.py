
"""
Responder node.

Handles:
- normal conversation
- AWS case study/blog RAG
"""

import logfire

from app.gateway.portkey_gateway import generate



SYSTEM_PROMPT = (
    "You are AWS Knowledge Assistant, an AI assistant specialized "
    "in AWS case studies and technical blog knowledge. "
    "For normal conversation, respond naturally and briefly. "
    "If the user asks who you are, what you are, or asks about "
    "your identity, say that you are the AWS Knowledge Assistant. "
    "Never identify yourself as ChatGPT, an OpenAI assistant, "
    "or by the name of the underlying LLM. "
    "Do not claim to be AWS or an AWS employee. "
    "For AWS questions, answer ONLY using the provided context. "
    "The context comes from AWS case studies and technical blog articles. "
    "If the provided context does not contain the answer, "
    "say you don't know. "
    "Never make up information."
)




def build_context(chunks: list[dict]) -> str:
    """Build context from retrieved Qdrant chunks."""

    blocks = []

    for i, chunk in enumerate(chunks, start=1):

        source = chunk.get(
            "source",
            "Unknown source",
        )

        text = chunk.get(
            "text",
            "",
        )

        blocks.append(
            f"[Source {i}: {source}]\n{text}"
        )

    return "\n\n".join(blocks)


def add_message(
    state: dict,
    role: str,
    content: str,
) -> None:
    """Append a message to the conversation history."""

    messages = state.get("messages", [])

    messages.append(
        {
            "role": role,
            "content": content,
        }
    )

    # Keep only the recent conversation.
    state["messages"] = messages[-10:]


def respond_node(state: dict) -> dict:

    question = state["question"]

    with logfire.span(
        "rag.responder",
        question=question,
    ):

        # ---------------------------------------------------------
        # Normal conversation
        # ---------------------------------------------------------

        if state.get("intent") == "conversation":

            logfire.info(
                "Generating conversational response"
            )

            response = generate(
                SYSTEM_PROMPT,
                question,
            )

            state["answer"] = response

            add_message(
                state,
                "user",
                question,
            )

            add_message(
                state,
                "assistant",
                response,
            )

            state["steps"] = state.get("steps", []) + [
                {
                    "node": "responder",
                    "output": "Generated conversational response",
                }
            ]

            logfire.info(
                "Conversational response generated",
                response_length=len(response),
            )

            return state

        # ---------------------------------------------------------
        # RAG response
        # ---------------------------------------------------------

        chunks = state.get(
            "retrieved_chunks",
            []
        )

        logfire.info(
            "Generating RAG response",
            retrieved_chunks=len(chunks),
        )

        # ---------------------------------------------------------
        # No retrieved context, or the grader marked it irrelevant
        # ---------------------------------------------------------

        if not chunks or state.get("grade") == "irrelevant":

            response = (
                "I couldn't find relevant information "
                "in the AWS case studies and blogs."
            )

            state["answer"] = response

            add_message(
                state,
                "user",
                question,
            )

            add_message(
                state,
                "assistant",
                response,
            )

            state["steps"] = state.get("steps", []) + [
                {
                    "node": "responder",
                    "output": (
                        "No relevant context found"
                        if not chunks
                        else "Retrieved context graded irrelevant"
                    ),
                }
            ]

            logfire.info(
                "No usable retrieved context available",
                grade=state.get("grade"),
            )

            return state

        # ---------------------------------------------------------
        # Build context
        # ---------------------------------------------------------

        context = build_context(chunks)

        logfire.info(
            "Context built",
            context_length=len(context),
            source_count=len(chunks),
        )

        # ---------------------------------------------------------
        # RAG prompt
        # ---------------------------------------------------------

        prompt = f"""
Answer the user's question using ONLY the retrieved
AWS case studies and technical blog context below.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Rules:
- Use only the retrieved context.
- Give a clear and useful answer, not just a one-line fact. Where the
  context supports it, briefly explain the "why"/"how" behind the
  answer (the reasoning, benefit, or mechanism), not just the name of
  the service or result.
- Prefer information directly supported by the context.
- Do not invent facts, and do not add reasoning the context doesn't
  support -- if the context only states the fact with no elaboration,
  give the fact plainly rather than speculating about the "why".
- Do not use outside AWS knowledge.
- If the context does not contain enough information to answer,
  say:
  "I couldn't find this information in the retrieved context."
"""

        # ---------------------------------------------------------
        # LLM generation
        # ---------------------------------------------------------

        logfire.info(
            "Calling response LLM",
            prompt_length=len(prompt),
        )

        response = generate(
            SYSTEM_PROMPT,
            prompt,
        )

        state["answer"] = response

        # ---------------------------------------------------------
        # Save conversation
        # ---------------------------------------------------------

        add_message(
            state,
            "user",
            question,
        )

        add_message(
            state,
            "assistant",
            response,
        )

        state["steps"] = state.get("steps", []) + [
            {
                "node": "responder",
                "output": "Generated answer from retrieved context",
            }
        ]

        logfire.info(
            "RAG response generated",
            response_length=len(response),
        )

        return state