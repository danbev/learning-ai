from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import ShellTool

from dotenv import load_dotenv 
load_dotenv()

shell_tool = ShellTool()
print(shell_tool.run({"commands": ["echo 'Bajja!'"]}))
print(f'{shell_tool.description=}')
print(f'{shell_tool.args=}')

shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
    "{", "{{"
).replace("}", "}}")
print(f'{shell_tool.description=}')

llm = ChatOpenAI(temperature=0)

self_ask_with_search = initialize_agent(
    [shell_tool], llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
blogs = self_ask_with_search.run(
    "Download the https://www.trustification.io/blog webpage and grep for blog urls. Return only a sorted list of them. Be sure to use double quotes."
)

print(blogs)
