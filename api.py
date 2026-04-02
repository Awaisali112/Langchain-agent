import os
import sqlite3
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from auth import router as auth_router, get_current_user

from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_tool_call,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    ModelFallbackMiddleware,
    SummarizationMiddleware,
)
from langchain.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain_community.tools import (
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
    ArxivQueryRun,
)
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "minimax-m2.7:cloud")
MODEL_TEMP = float(os.getenv("MODEL_TEMP", "0.7"))
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "research_research.db")

# ── Tools ──────────────────────────────────────────────────────────────────────
ddgs_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
search_tool = DuckDuckGoSearchResults(
    api_wrapper=ddgs_wrapper,
    name="web_search",
    description="Search the internet for real-time information.",
)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_wrapper,
    name="wikipedia",
    description="Search Wikipedia for well-established, encyclopedic knowledge.",
)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
arxiv_tool = ArxivQueryRun(
    api_wrapper=arxiv_wrapper,
    name="Arxiv",
    description="Search arXiv for peer-reviewed academic papers and preprints.",
)


@tool
def get_current_datetime():
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Middleware ─────────────────────────────────────────────────────────────────
@wrap_tool_call
def handle_tool_call_error(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return f"Tool error: {str(e)}"


tool_retry = ToolRetryMiddleware(
    max_retries=2, tools=["search_tool"], on_failure="continue",
    max_delay=60, backoff_factor=1.5,
)
model_retry = ModelRetryMiddleware(
    max_retries=2, on_failure="continue", max_delay=60, backoff_factor=2.0,
)
model_fallback = ModelFallbackMiddleware("ollama:minimax-m2.7:cloud")
summ_middleware = SummarizationMiddleware(
    model="ollama:minimax-m2.7:cloud",
    trigger=("tokens", 4000),
    keep=("messages", 20),
)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""
You are a precise and thorough research assistant. Your job is to investigate topics deeply and return well-structured, accurate answers grounded in real sources.

## IMPORTANT:
Today is {datetime.today()}

## Your Tools
- **web_search** — use for current events, news, and anything time-sensitive
- **wikipedia** — use for definitions, background context, and established facts
- **arxiv** — use for academic papers, technical research, and scientific claims
- **get_current_datetime** — use when the user asks about the current time or date

## How You Work
1. Analyze the user's question and identify what kind of information is needed.
2. Choose the right tool(s) — you may call multiple tools if needed.
3. Synthesize results into a clear, structured response.
4. Always cite where information came from (web, Wikipedia, paper title, etc.).
5. If a tool returns an error or empty result, try rephrasing the query or use a different tool.

## Rules
- Never fabricate facts, citations, or paper titles.
- If you don't know something and can't find it via tools, say so clearly.
- Keep responses focused — don't pad with filler.
- For technical topics, prefer arxiv over web_search.
- For recent events (< 1 year), always use web_search.
"""

# ── Agent factory (singleton) ──────────────────────────────────────────────────
_agent = None
_db_conn = None


def get_agent():
    global _agent, _db_conn
    if _agent is None:
        llm = ChatOllama(model=MODEL_NAME, temperature=MODEL_TEMP)
        _db_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        memory = SqliteSaver(conn=_db_conn)
        _agent = create_agent(
            model=llm,
            tools=[search_tool, wiki_tool, arxiv_tool, get_current_datetime],
            system_prompt=SYSTEM_PROMPT,
            middleware=[handle_tool_call_error, tool_retry, model_retry, model_fallback, summ_middleware],
            checkpointer=memory,
            name="Research Assistant",
        )
    return _agent


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    tool_permissions: dict[str, bool] = {
        "web_search": True,
        "wikipedia": True,
        "Arxiv": True,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/chat")
async def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Stream agent response as Server-Sent Events."""

    allowed = req.tool_permissions

    # Per-request HITL middleware using the permissions from the request body
    @wrap_tool_call
    def request_hitl(request, handler):
        tool_name = request.tool.name
        if tool_name in ["web_search", "wikipedia", "Arxiv"]:
            if not allowed.get(tool_name, True):
                return f"'{tool_name}' is disabled by user settings."
        return handler(request)

    agent = get_agent()
    config = {"configurable": {"thread_id": req.thread_id}}

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def run_agent():
            try:
                for chunk in agent.stream(
                    {"messages": [HumanMessage(content=req.message)]},
                    config=config,
                    stream_mode="values",
                ):
                    latest = chunk["messages"][-1]
                    if isinstance(latest, AIMessage) and latest.content:
                        loop.call_soon_threadsafe(
                            queue.put_nowait, {"type": "text", "content": latest.content}
                        )
                    elif hasattr(latest, "tool_calls") and latest.tool_calls:
                        tool_names = [tc["name"] for tc in latest.tool_calls]
                        loop.call_soon_threadsafe(
                            queue.put_nowait, {"type": "tool_call", "tools": tool_names}
                        )
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"type": "error", "content": str(e)}
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        loop.run_in_executor(None, run_agent)

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
