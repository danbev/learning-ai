from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import ShellTool

from dotenv import load_dotenv 
load_dotenv()

shell_tool = ShellTool()

llm = ChatOpenAI(temperature=0)

shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
    "{", "{{"
).replace("}", "}}")

self_ask_with_search = initialize_agent(
    [shell_tool], llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
blogs = self_ask_with_search.run(
    "Download the https://www.trustification.io/blog webpage and grep for blog urls. Return only a sorted list of them. Be sure to use double quotes."
)

print(blogs)
