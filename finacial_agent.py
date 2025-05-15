from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

## web search agent
web_search_agent=Agent(
    name="web search agent",
    role="Search the web for the information",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=["Always include the source"],
    show_tool_calls=True,
    markdown=True,
)

## Financial agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-70b-8192"),
    tools=[
        YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True,company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent=Agent(
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    team=[web_search_agent,finance_agent],
    instructions=["Always include sources","Use table to display the data"],
    show_tool_calls=True,
    markdown=True,
)
multi_ai_agent.print_response("Summarize analyst recomendation and share the letest news for NVDA",stream=True)