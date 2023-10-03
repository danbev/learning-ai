import streamlit as st
import openai
import os, random, time
import requests

from langchain import OpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.agents import load_tools, Tool, initialize_agent, AgentType
from langchain.requests import RequestsWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import JSONLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

from dotenv import load_dotenv 
load_dotenv()

vex_persist_directory = 'chroma/trust/vex'
cve_persist_directory = 'chroma/trust/cve'
embedding = OpenAIEmbeddings()

def load_vex_docs():
    vec_loader = JSONLoader(file_path='./src/vex-stripped.json', jq_schema='.document', text_content=False)
    vex_docs = vec_loader.load()
    #print(f'Pages: {len(docs)}, type: {type(docs[0])})')
    #print(f'{docs[0].metadata}')

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len  # function used to measure chunk size
    )
    vex_splits = r_splitter.split_documents(vex_docs)
    print(f'vex_splits len: {len(vex_splits)}, type: {type(vex_splits[0])}')
    vex_vectorstore = Chroma.from_documents(
        documents=vex_splits,
        embedding=embedding,
        persist_directory=vex_persist_directory
    )
    vex_vectorstore.persist()

#load_vex_docs()

llm = OpenAI(temperature=0)

vex_vectorstore = Chroma(persist_directory=vex_persist_directory, embedding_function=embedding)
vex_retriever = vex_vectorstore.as_retriever(search_kwargs={'k': 3})
vex_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vex_retriever, verbose=True
)

#tools = load_tools(["google-serper", "llm-math"], llm=llm)
tools = load_tools(["llm-math"], llm=llm)
tools.append(Tool(name="VEX", func=vex_chain.run, description="useful for when you need to answer questions about the VEX documents which are security advisories in the format RHSA-XXXX:XXXX, where X can be any number."))

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)

# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    memory=memory
)
#print(agent)
#exit()

#agent_executor = initialize_agent(
#    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#    tools=tools,
#    llm=llm,
#    verbose=True,
#    max_iterations=3,
#    early_stopping_method='generate',
#    memory=memory
#)

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
