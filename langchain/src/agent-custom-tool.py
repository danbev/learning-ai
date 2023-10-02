import os, random
from langchain import OpenAI

from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.agents import load_tools, Tool, initialize_agent, AgentType

from langchain.requests import RequestsWrapper
import requests

from dotenv import load_dotenv 
load_dotenv()

def vex(input):
    token_endpoint = 'http://localhost:8090/realms/chicken/protocol/openid-connect/token'
    client_id = 'walker'
    client_secret = os.getenv("CLIENT_SECRET")
    access_token = []

    response = requests.post(
        token_endpoint,
        data={
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret,
        }
    )
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data['access_token']
    else:
        print(f"Failed to obtain access token. Status code: {response.status_code}, Response: {response.text}")
        exit()

    headers = {"Authorization": f"Bearer {access_token}"}
    requests_wrapper = RequestsWrapper(headers=headers)
    return requests_wrapper.get(f'http://localhost:8081/api/v1/vex?advisory=${input}')

vex_tool = Tool(
    name='VEX',
    func= vex,
    description="Useful when you need to get information related to a VEX using its advisory ID. An example of an advisory ID is RHSA-2023:1441"
)

#llm = OpenAI(model_name="gpt-3.5-turbo" ,temperature=0)
llm = OpenAI(model_name="text-davinci-003" ,temperature=0)

tools = load_tools(["google-serper", "llm-math"], llm=llm)

def random_num(input=""):
    print(f'In random_num function. input: {input}')
    return random.randint(0,5)

random_tool = Tool(
    name='Random number',
    func= random_num,
    description="Useful for when you need to get a random number. input should be 'random'"
)
tools.append(random_tool)
tools.append(vex_tool)

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

result = agent_executor("What time is it in Brisbane?")
print(result['output'])

result = agent_executor("Can you show me a random number?")
print(result['output'])

result = agent_executor("Can you tell me what the VEX RHSA-2023:1441 is about?")
print(result['output'])

result = agent_executor("Which CVEs are related to the VEX RHSA-2023:1441?")
print(result['output'])

#print(agent_executor.agent.llm_chain.prompt.messages[0].prompt.template)
#agent_executor.agent.llm_chain.prompt.messages[0].prompt.template = updated_prompt
