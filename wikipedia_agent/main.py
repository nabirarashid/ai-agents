from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core .output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """You are a research assistant that helps generate research papers.
     
     You have access to tools to search for information and save content.
     Use the tools to gather comprehensive information about the user's query.
     
     After gathering information with tools, provide your final response in this exact format:
     {format_instructions}
     
     Steps to follow:
     1. Use search_tool to find current information
     2. Use wiki_tool for encyclopedic knowledge
     3. Use save_tool to save important findings
     4. Synthesize the information into the required format
     """),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,)

query =  input("What can I help you research today? ")
raw_response = agent_executor.invoke({"query": query})

try:
    output_text = raw_response["output"]

    # extracting the json part
    if "```json" in output_text:
        json_start = output_text.find("```json") + 7
        json_end = output_text.find("```", json_start)
        json_text = output_text[json_start:json_end].strip()
    else:
        json_text = output_text

    structured_response = parser.parse(json_text)
    print(f"Topic: {structured_response.topic}")
    print(f"Summary: {structured_response.summary}")
