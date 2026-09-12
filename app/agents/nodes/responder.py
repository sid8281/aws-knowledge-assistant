
"""
Responder node.

Handles:
- normal conversation
- AWS case study/blog RAG
- AWS official documentation via MCP
"""

import logfire

from app.gateway.portkey_gateway import generate


SYSTEM_PROMPT = (
    "You are AWS Knowledge Assistant, an AI assistant specialized "
    "in AWS case studies, technical blogs, and official AWS documentation. "
    "For normal conversation, respond naturally and briefly. "
    "If the user asks who you are, what you are, or asks about "
    "your identity, say that you are the AWS Knowledge Assistant. "
    "Never identify yourself as ChatGPT, an OpenAI assistant, "
    "or by the name of the underlying LLM. "
    "Do not claim to be AWS or an AWS employee. "
    "For RAG questions, use only the provided AWS case-study and "
    "technical-blog context. "
    "For MCP questions, use only the provided official AWS "
    "documentation context. "
    "Never make up information."
)


def build_context(chunks: list[dict]) -> str:
    """Build context from retrieved Qdrant/MCP chunks."""

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
        "responder",
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
        # Determine whether this is MCP or RAG
        # ---------------------------------------------------------

        is_mcp = state.get("intent") == "mcp"

        # ---------------------------------------------------------
        # Get retrieved context
        #
        # For RAG:
        # retrieved_chunks = Qdrant/BM25/etc. results
        #
        # For MCP:
        # retrieved_chunks should contain AWS documentation
        # returned by the AWS Knowledge MCP fallback node.
        # ---------------------------------------------------------

        chunks = state.get(
            "retrieved_chunks",
            []
        )

        logfire.info(
            "Preparing responder context",
            intent=state.get("intent"),
            retrieved_chunks=len(chunks),
        )

        # ---------------------------------------------------------
        # No context available
        # ---------------------------------------------------------

        if not chunks:

            if is_mcp:
                response = (
                    "I couldn't find enough information in the "
                    "AWS documentation retrieved through the "
                    "AWS Knowledge MCP."
                )

            else:
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
                        "No AWS documentation context found"
                        if is_mcp
                        else "No relevant RAG context found"
                    ),
                }
            ]

            logfire.info(
                "No usable context available",
                intent=state.get("intent"),
                grade=state.get("grade"),
            )

            return state

        # ---------------------------------------------------------
        # Grader rejection
        #
        # This applies to normal RAG.
        #
        # MCP documentation should not normally reach this branch
        # with an RAG relevance grade.
        # ---------------------------------------------------------

        if not is_mcp and state.get("grade") == "irrelevant":

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
                    "output": "Retrieved context graded irrelevant",
                }
            ]

            logfire.info(
                "RAG context graded irrelevant",
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
            intent=state.get("intent"),
        )

        # ---------------------------------------------------------
        # MCP prompt
        # ---------------------------------------------------------

        if is_mcp:

            prompt = f"""
Answer the user's question using ONLY the official AWS
documentation context retrieved through the AWS Knowledge MCP.

USER QUESTION:
{question}

AWS DOCUMENTATION CONTEXT:
{context}

Rules:
- Use only the provided AWS documentation context.
- Give practical and accurate instructions.
- Do not invent AWS commands, configuration values,
  parameters, or steps.
- When the documentation contains relevant information,
  explain the steps clearly.
- Prefer official AWS terminology.
- Do not use information from the local AWS case-study/blog
  knowledge base.
- If the documentation does not contain enough information,
  say:
  "I couldn't find enough information in the AWS documentation
  retrieved for this question."
"""

            step_output = (
                "Generated answer from official AWS documentation "
                "via MCP"
            )

        # ---------------------------------------------------------
        # RAG prompt
        # ---------------------------------------------------------

        else:

            prompt = f"""
Answer the user's question using ONLY the retrieved
AWS case studies and technical blog context below.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Rules:
- Use only the retrieved context.
- Give a clear and useful answer.
- Prefer information directly supported by the context.
- Do not invent facts.
- Do not use outside AWS knowledge.
- If the context does not contain enough information,
  say:
  "I couldn't find this information in the retrieved context."
"""

            step_output = (
                "Generated answer from retrieved AWS "
                "case-study/blog context"
            )

        # ---------------------------------------------------------
        # LLM generation
        # ---------------------------------------------------------

        logfire.info(
            "Calling response LLM",
            prompt_length=len(prompt),
            intent=state.get("intent"),
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
                "output": step_output,
            }
        ]

        logfire.info(
            "Response generated",
            response_length=len(response),
            intent=state.get("intent"),
        )

        return state
