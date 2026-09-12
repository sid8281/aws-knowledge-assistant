"""
Streamlit chat UI for the AWS Knowledge Assistant.

Run:
    streamlit run ui/streamlit_app.py
"""

import html
import os
import time
import uuid

import requests
import streamlit as st


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_URL = f"{API_BASE_URL}/query"
API_KEY = os.getenv("API_KEY", "")

st.set_page_config(
    page_title="AWS Knowledge Assistant",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "request_count" not in st.session_state:
    st.session_state.request_count = 0

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# -------------------------------------------------------------------
# Styling — dark theme, AWS-orange accent
# -------------------------------------------------------------------

st.markdown(
    """
    <style>
        :root {
            --aws-orange: #FF9900;
            --bg-app: #0e1117;
            --bg-panel: #161b22;
            --bg-card: #1c2128;
            --border: #2d333b;
            --text-primary: #e6edf3;
            --text-muted: #8b949e;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background-color: var(--bg-app);
        }

        [data-testid="stSidebar"] {
            background-color: var(--bg-panel);
            border-right: 1px solid var(--border);
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 6rem;
        }

        body,
        
        /* Fix code blocks */
        [data-testid="stCode"],
        [data-testid="stCode"] pre,
        [data-testid="stCode"] code {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        }

        [data-testid="stCode"] pre {
        border: 1px solid #2d333b !important;
        border-radius: 0.6rem !important;
        }

        [data-testid="stCode"] code {
        color: #e6edf3 !important;
        }

        /* Header */
        .app-header {
            padding: 0 0 1.5rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1.5rem;
        }

        .app-title {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.2rem;
        }

        .app-title .accent {
            color: var(--aws-orange);
        }

        .app-subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        /* Status cards */
        .status-card {
            padding: 0.65rem 0.8rem;
            border-radius: 0.6rem;
            border: 1px solid var(--border);
            background: var(--bg-card);
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-primary);
        }

        .status-dot {
            color: #3fb950;
            font-weight: bold;
        }

        .status-card small {
            color: var(--text-muted);
        }

        /* Source cards */
        .source-card {
            padding: 0.8rem;
            border: 1px solid var(--border);
            border-radius: 0.6rem;
            margin-bottom: 0.5rem;
            background: var(--bg-card);
        }

        .source-title {
            font-weight: 600;
            margin-bottom: 0.25rem;
            color: var(--text-primary);
        }

        .source-path {
            color: var(--text-muted);
            font-size: 0.78rem;
            word-break: break-word;
        }

        /* Pipeline trace */
        .pipeline {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            flex-wrap: wrap;
            margin-top: 0.4rem;
        }

        .pipeline-node {
            padding: 0.3rem 0.55rem;
            border-radius: 0.45rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            font-size: 0.78rem;
            color: var(--text-primary);
        }

        .pipeline-arrow {
            color: var(--text-muted);
        }

        /* Welcome box */
        .welcome {
            padding: 1.5rem 1.75rem;
            border: 1px solid var(--border);
            border-radius: 0.9rem;
            background: var(--bg-panel);
            margin: 1rem 0 1.5rem 0;
            color: var(--text-primary);
        }

        /* Blocked-answer banner */
        .blocked-banner {
            padding: 0.9rem 1.1rem;
            border-radius: 0.6rem;
            border: 1px solid #d29922;
            background: rgba(210, 153, 34, 0.12);
            color: #e3b341;
            margin-bottom: 0.6rem;
            font-size: 0.9rem;
        }

        /* Chat bubbles */
        [data-testid="stChatMessage"] {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 0.8rem;
            padding: 0.4rem 0.2rem;
            margin-bottom: 0.5rem;
        }

        /* Buttons */
        .stButton > button {
            background-color: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
        }

        .stButton > button:hover {
            border-color: var(--aws-orange);
            color: var(--aws-orange);
        }

        /* Suggestion buttons */
        div[data-testid="stVerticalBlock"] .stButton > button {
            text-align: left;
            justify-content: flex-start;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def start_new_conversation():
    """Reset the frontend and backend conversation state."""
    st.session_state.messages = []
    st.session_state.request_count = 0
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pending_question = None


def render_sources(sources):
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">📄 Source {index}</div>
                    <div class="source-path">{html.escape(str(source))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_steps(steps):
    if not steps:
        return

    with st.expander("🔍 Retrieval pipeline"):
        nodes = [step.get("node", "unknown") for step in steps]

        pipeline_html = '<div class="pipeline">'

        for index, node in enumerate(nodes):
            safe_node = html.escape(str(node))
            pipeline_html += f'<span class="pipeline-node">{safe_node}</span>'

            if index < len(nodes) - 1:
                pipeline_html += '<span class="pipeline-arrow">→</span>'

        pipeline_html += "</div>"

        st.markdown(pipeline_html, unsafe_allow_html=True)
        st.divider()

        for step in steps:
            st.write(
                f"**{step.get('node', 'unknown')}**  \n"
                f"{step.get('output', '')}"
            )


def render_welcome():
    if st.session_state.messages or st.session_state.pending_question:
        return

    st.markdown(
        """
        <div class="welcome">
            Ask questions about AWS case studies and technical blogs.
            The assistant uses hybrid retrieval, reranking, query
            rewriting, conversational memory, and guardrails —
            with a live AWS documentation fallback when the local
            knowledge base doesn't have an answer.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Try asking")

    suggestions = [
        "What AWS services does Netflix use?",
        "Which AWS service does Slack use for its database?",
        "What AWS services are common between Netflix and Slack?",
    ]

    cols = st.columns(len(suggestions))

    for col, suggestion in zip(cols, suggestions):
        with col:
            if st.button(
                f"💡 {suggestion}",
                key=f"suggestion-{suggestion}",
                use_container_width=True,
            ):
                st.session_state.pending_question = suggestion


def ask_backend(question: str) -> tuple[dict, float]:
    """Send a question to the API. Returns (response_data, latency_seconds)."""

    start_time = time.perf_counter()

    try:
        response = requests.post(
            API_URL,
            json={
                "question": question,
                "thread_id": st.session_state.thread_id,
            },
            headers={"X-API-Key": API_KEY},
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()

    except requests.exceptions.Timeout:
        data = {
            "answer": "The request timed out. Please try again.",
            "sources": [],
            "steps": [],
            "blocked": False,
        }

    except requests.exceptions.HTTPError as error:
        if error.response is not None and error.response.status_code == 401:
            answer = (
                "Authentication failed: the API_KEY this app is using "
                "doesn't match what the backend expects. Check the "
                "API_KEY environment variable on both sides."
            )
        else:
            status = (
                error.response.status_code
                if error.response is not None
                else "unknown"
            )
            answer = f"The backend returned an error (HTTP {status})."

        data = {
            "answer": answer,
            "sources": [],
            "steps": [],
            "blocked": False,
        }

    except requests.exceptions.RequestException as error:
        data = {
            "answer": f"Unable to reach the backend: {error}",
            "sources": [],
            "steps": [],
            "blocked": False,
        }

    except Exception as error:
        data = {
            "answer": f"Unexpected error: {error}",
            "sources": [],
            "steps": [],
            "blocked": False,
        }

    latency = time.perf_counter() - start_time

    return data, latency


def render_assistant_message(message: dict):
    """Render one assistant message from session_state.messages.

    This is the single place that knows how to draw an assistant
    reply (answer/blocked banner + sources + steps + latency) so the
    chat-history loop below never has its own separate copy of this
    logic to drift out of sync with.
    """

    answer = message.get("content", "No answer returned.")

    if message.get("blocked"):
        st.markdown(
            f'<div class="blocked-banner">🛡️ {html.escape(str(answer))}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(answer)

    render_sources(message.get("sources", []))
    render_steps(message.get("steps", []))

    latency = message.get("latency")

    if latency is not None:
        st.caption(f"⚡ Response time: {latency:.2f}s")


def handle_question(question: str):
    """Send a question, showing the user bubble immediately and the
    assistant bubble as soon as the backend responds, instead of
    waiting for the whole round-trip before anything renders."""

    question = question.strip()

    if not question:
        return

    st.session_state.request_count += 1

    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching AWS knowledge base..."):
            data, latency = ask_backend(question)

        assistant_message = {
            "role": "assistant",
            "content": data.get("answer", "No answer returned."),
            "sources": data.get("sources", []),
            "steps": data.get("steps", []),
            "blocked": data.get("blocked", False),
            "latency": latency,
        }

        render_assistant_message(assistant_message)

    st.session_state.messages.append(assistant_message)


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ☁️ AWS RAG")
    st.caption("AWS Knowledge Assistant")

    st.divider()

    if st.button("＋ New conversation", use_container_width=True):
        start_new_conversation()
        st.rerun()

    st.divider()

    st.markdown("### System")

    st.markdown(
        """
        <div class="status-card">
            <span class="status-dot">●</span> API
            <br><small>FastAPI backend</small>
        </div>
        <div class="status-card">
            <span class="status-dot">●</span> Retrieval
            <br><small>Hybrid search + reranker</small>
        </div>
        <div class="status-card">
            <span class="status-dot">●</span> CRAG Grading
            <br><small>Relevance check + retry</small>
        </div>
        <div class="status-card">
            <span class="status-dot">●</span> AWS Docs Fallback
            <br><small>Live docs via MCP</small>
        </div>
        <div class="status-card">
            <span class="status-dot">●</span> Memory
            <br><small>LangGraph checkpointing</small>
        </div>
        <div class="status-card">
            <span class="status-dot">●</span> Guardrails
            <br><small>NeMo Guardrails</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Architecture")
    st.caption(
        "Planner → Query Rewriter → Hybrid Retrieval → "
        "CRAG Grader → (AWS Docs Fallback) → Responder"
    )

    st.divider()

    st.caption(f"Questions asked: {st.session_state.request_count}")
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}…`")


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">
            AWS <span class="accent">Knowledge</span> Assistant
        </div>
        <div class="app-subtitle">
            AWS case studies & technical knowledge, powered by RAG
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Welcome
# -------------------------------------------------------------------

render_welcome()


# -------------------------------------------------------------------
# Chat history
# -------------------------------------------------------------------

for message in st.session_state.messages:
    role = message["role"]

    with st.chat_message(role):
        if role == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message["content"])


# -------------------------------------------------------------------
# Chat input
# -------------------------------------------------------------------

# Always render the chat input.
# This prevents the input box from disappearing when a suggestion
# button is clicked.
typed_question = st.chat_input(
    "Ask a question about AWS case studies..."
)


# -------------------------------------------------------------------
# Question handling
# -------------------------------------------------------------------

pending_question = st.session_state.pending_question

if pending_question:
    st.session_state.pending_question = None

# A clicked suggestion gets priority over typed input.
question = pending_question or typed_question

if question:
    handle_question(question)

    # Rerun once so the newly stored messages are rendered
    # through the normal chat-history section.
    st.rerun()
