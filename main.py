import os
import sys
from dotenv import load_dotenv
from datetime import datetime
import sqlite3

# LangChain
from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_tool_call,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    ModelFallbackMiddleware,
    SummarizationMiddleware
)
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool
from langchain_community.tools import (
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
    ArxivQueryRun
)
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
    ArxivAPIWrapper
)

# LangGraph
from langgraph.checkpoint.sqlite import SqliteSaver

# Ollama Model
from langchain_ollama import ChatOllama

# Load Environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_TEMP = float(os.getenv("MODEL_TEMP", "0.7"))
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "research_assistant.db")

# Tool wrappers
ddgs_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
search_tool = DuckDuckGoSearchResults(
    api_wrapper=ddgs_wrapper,
    name="web_search",
    description="Search the internet for real-time information."
)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_wrapper,
    name="wikipedia",
    description="Search Wikipedia for well-established, encyclopedic knowledge."
)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
arxiv_tool = ArxivQueryRun(
    api_wrapper=arxiv_wrapper,
    name="Arxiv",
    description="Search arXiv for peer-reviewed academic papers and preprints."
)

@tool
def get_current_datetime():
    """Get the current date and time."""
    now_datetime = datetime.now()
    return now_datetime.strftime("%Y-%m-%d %H:%M:%S")


# ---------------- Human-in-the-loop middleware ----------------
tool_permissions = {}  # global dictionary to store yes/no permissions

def ask_permissions():
    """Ask permission for all three search tools once at startup."""
    global tool_permissions
    tool_permissions = {}  # reset

    tools_to_prompt = ["web_search", "wikipedia", "Arxiv"]
    for tool_name in tools_to_prompt:
        while True:
            ans = input(f"Allow '{tool_name}'? (yes/no): ").strip().lower()
            if ans in ["yes", "y", "no", "n"]:
                tool_permissions[tool_name] = ans in ["yes", "y"]
                break
            print("Please answer yes/no")

@wrap_tool_call
def human_in_loop(request, handler):
    tool_name = request.tool.name

    if tool_name in ["web_search", "wikipedia", "Arxiv"]:
        if not tool_permissions.get(tool_name, False):
            return f"'{tool_name}' blocked by user"

    return handler(request)


# ---------------- Other Middleware ----------------
@wrap_tool_call
def handle_tool_call_error(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return f"Tool error: {str(e)}"

tool_retry = ToolRetryMiddleware(
    max_retries=2,
    tools=["search_tool"],
    on_failure="continue",
    max_delay=60,
    backoff_factor=1.5
)

model_retry = ModelRetryMiddleware(
    max_retries=2,
    on_failure="continue",
    max_delay=60,
    backoff_factor=2.0
)

model_fallback = ModelFallbackMiddleware(
    "ollama:minimax-m2.5:cloud",
)

summ_middleware = SummarizationMiddleware(
    model="ollama:minimax-m2.5:cloud",
    trigger=("tokens", 4000),
    keep=("messages", 20)
)

# ---------------- Tools ----------------
custom_tools = [search_tool, wiki_tool, arxiv_tool, get_current_datetime]

# ---------------- System Prompt ----------------
custom_system_prompt = f"""
You are a precise and thorough research assistant. Your job is to investigate topics deeply and return well-structured, accurate answers grounded in real sources.

## IMPORTANT:
Today is {datetime.today()}
Now is {datetime.now()}

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

# ---------------- Agent Setup ----------------
def run_research_agent():
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=MODEL_TEMP
    )

    db_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    memory = SqliteSaver(conn=db_conn)

    tool_middleware = [
        human_in_loop,           # HITL
        handle_tool_call_error,
        tool_retry,
        model_retry,
        model_fallback,
        summ_middleware
    ]

    agent = create_agent(
        model=llm,
        tools=custom_tools,
        system_prompt=custom_system_prompt,
        middleware=tool_middleware,
        checkpointer=memory,
        name="Research Assistant"
    )

    return agent



# ---------------- Stream Response ----------------
def stream_response(agent, query: str , config: dict):
    for chunk in agent.stream({"messages": [HumanMessage(content=query)]}, config=config, stream_mode="values"):
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            if isinstance(latest_message, AIMessage):
                print(f"Agent: {latest_message.content}")
        elif latest_message.tool_calls:
            print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")

# ---------------- Banner ----------------

def banner():
    R  = "\033[0m"
    # Cyan/Blue ramp
    C1 = "\033[38;2;0;212;255m"
    C2 = "\033[38;2;0;184;224m"
    C3 = "\033[38;2;0;156;192m"
    C4 = "\033[38;2;0;127;160m"
    C5 = "\033[38;2;0;170;212m"
    # Green ramp
    G1 = "\033[38;2;0;255;136m"
    G2 = "\033[38;2;0;224;122m"
    G3 = "\033[38;2;0;192;106m"
    G4 = "\033[38;2;0;160;88m"
    G5 = "\033[38;2;0;200;120m"
    # Accents
    AM = "\033[38;2;255;184;0m"   # amber
    DM = "\033[38;2;30;60;80m"    # dim
    MT = "\033[38;2;60;100;120m"  # muted

    print(f"""
{C1}██╗      █████╗ ███╗   ██╗ ██████╗  ██████╗██╗  ██╗ █████╗ ██╗███╗   ██╗{R}
{C2}██║     ██╔══██╗████╗  ██║██╔════╝ ██╔════╝██║  ██║██╔══██╗██║████╗  ██║{R}
{C3}██║     ███████║██╔██╗ ██║██║  ███╗██║     ███████║███████║██║██╔██╗ ██║{R}
{C4}██║     ██╔══██║██║╚██╗██║██║   ██║██║     ██╔══██║██╔══██║██║██║╚██╗██║{R}
{C5}███████╗██║  ██║██║ ╚████║╚██████╔╝╚██████╗██║  ██║██║  ██║██║██║ ╚████║{R}
{C1}╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝{R}

{G1} █████╗  ██████╗ ███████╗███╗   ██╗████████╗{R}
{G2}██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝{R}
{G3}███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║{R}
{G4}██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║{R}
{G5}██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║{R}
{G1}╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝{R}

{DM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{R}
  {C1}⟡ web_search{R}   {C2}⟡ wikipedia{R}   {G1}⟡ arxiv{R}   {AM}⟡ datetime{R}
  {MT}● STATUS{R} {G1}ONLINE{R}    {MT}● MEMORY{R} SQLite    {MT}● MODEL{R} Ollama / Local LLM
{DM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{R}
  {DM}type your query  ·  {C1}exit{R}{DM} / {C1}quit{R}{DM} / {C1}q{R}{DM} to exit{R}
""")

# ---------------- Main ----------------
def main():
    banner()
    ask_permissions()  # Ask for all three search tool permissions once
    agent = run_research_agent()
    config = {"configurable": {"thread_id": "my_bucket"}}

    while True:
        try:
            query = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\nGood Bye!") 
            sys.exit(0)

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Good Bye! Thanks for searching!")
            sys.exit(0)

        try:
            stream_response(agent, query, config)    
        except Exception as err:
            print(f"Error occurred: {err}")

# ---------------- Run ----------------
if __name__ == "__main__":
    main()
