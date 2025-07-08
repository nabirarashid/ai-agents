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
         """You are a research assistant that will help generate a research paper.
         Answer the user query and use the necessary tools.
         You must wrap the output in this format and provide no other text\n{format_instructions}"""),
         ("human", "{query}"),
        ("ai", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool]
agent = create_tool_calling_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query =  input("What can I help you research today? ")
raw_response = agent_executor.invoke({"query": query})

print(raw_response)

# try:
#     structured_response = parser.parse(raw_response.get("output"))
#     print(structured_response)

# except Exception as e:
#     print(f"Error parsing response: {e}. Raw response was: {raw_response}")
