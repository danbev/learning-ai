import os, random
from langchain import OpenAI

from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.agents import load_tools, Tool, initialize_agent, AgentType

from dotenv import load_dotenv 
load_dotenv()

#llm = OpenAI(model_name="gpt-3.5-turbo" ,temperature=0)
llm = OpenAI(model_name="text-davinci-003" ,temperature=0)

tools = load_tools(["google-serper", "llm-math"], llm=llm)

def bajja(input=""):
    return 'Bajja is a silly thing that is sometimes put in debug messages when I dont know what to write.'

bajja_tool = Tool(
    name='Bajja',
    func= bajja,
    description="Useful for when you need to answer questions about Bajja. input should be bajja "
)

def random_num(input=""):
    print(f'In random_num function. input: {input}')
    return random.randint(0,5)

random_tool = Tool(
    name='Random number',
    func= random_num,
    description="Useful for when you need to get a random number. input should be 'random'"
)
tools.append(bajja_tool)
tools.append(random_tool)

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)

agent_executor = initialize_agent(
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=memory
)

result = agent_executor("What time is it in Brisbase?")
print(result['output'])

result = agent_executor("Can you show me a random number?")
print(result['output'])

result = agent_executor("What is the meaning of bajja?")
print(result['output'])

#print(agent_executor.agent.llm_chain.prompt.messages[0].prompt.template)
updated_prompt = """
Answer the following questions as best you can. You have access to the following tools:

google_serper: A low-cost Google Search API.Useful for when you need to answer questions about current events.Input should be a search query.
Calculator: Useful for when you need to answer questions about math.
Meaning of Life: Useful for when you need to answer questions about the meaning of life. input should be MOL
Random number: Useful for when you need to get a random number. input should be 'random'

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here). You don't know anything about random numbers or the meaning of life.

The only values that should be in the "action" field are: google_serper, Calculator, Meaning of Life, Random number

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Reminder to always use the exact characters `Final Answer` when responding.
"""
#agent_executor.agent.llm_chain.prompt.messages[0].prompt.template = updated_prompt

#result = agent_executor("What is the meaning of life?")
#print(result['output'])
