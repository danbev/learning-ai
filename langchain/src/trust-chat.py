import streamlit as st
import openai
import os, random, time
import requests
from langchain import OpenAI

from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.agents import load_tools, Tool, initialize_agent, AgentType
from langchain.requests import RequestsWrapper

from dotenv import load_dotenv 
load_dotenv()

def vex(input):
    print("VEX tool, input: ", input)
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

llm = OpenAI(model_name="text-davinci-003" ,temperature=0)

tools = load_tools(["google-serper", "llm-math"], llm=llm)
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

st.title("Trustification Chat UI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Trustification something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        response = agent_executor(prompt)
        full_response += response["output"]
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
