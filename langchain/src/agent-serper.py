import openai
import os
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from dotenv import load_dotenv 
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

llm = OpenAI(model_name="text-davinci-003" ,temperature=0)
tools = load_tools(["google-serper", "llm-math"], llm=llm)

# In this case the AgentType is CHAT_ZERO_SHOT_REACT_DESCRIPTION which is
# a zero-shot meaning that the complete context is provided by the agent and
# there is no memory of previous interactions. The REACT part is the ReAct,
# reason and act, part of the agent. The description indicates that the agent
# will trigger tools based on the description of the tool.
agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    

result = agent.run("Who is Austin Powers? What is his current age raised to the 0.23 power? Return his age in the result")
print(f'{result=}')
