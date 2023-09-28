import openai
import os
from langchain.llms import OpenAI
from langchain.agents import load_tools, Tool, initialize_agent, AgentType
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv 
load_dotenv()

def count_tokens(agent, query):
    with get_openai_callback() as cb:
        result = agent(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")


llm = OpenAI(model_name="text-davinci-003" ,temperature=0)
tools = load_tools(["google-serper", "llm-math"], llm=llm)

# The following prompt and llm_chain is to allow the agent to use the LLMChain
# for general purpose queries and logic that is was trained on.
prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='use this tool for general purpose queries and logic'
)
#tools.append(llm_tool)

# In this case the AgentType is CHAT_ZERO_SHOT_REACT_DESCRIPTION which is
# a zero-shot meaning there will not be multiple inter-dependant interactions
# so there is no memory of previous interactions.
# The REACT part is the ReAct, reason and act, part of the agent.
# The description indicates that the agent will trigger tools based on the
# description of the tool.
agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True,
                         max_iterations=3) # the default value is 15
    

result = count_tokens(agent,
                      "Who is Austin Powers? What is his current age raised to the 0.23 power? Return his age in the result")
print(f'{result["output"]}')
