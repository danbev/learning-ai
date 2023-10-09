import streamlit as st
import openai
import os, random, time
import requests

from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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
from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish

from dotenv import load_dotenv

from dotenv import load_dotenv 
load_dotenv()

vex_persist_directory = 'chroma/trust/vex'
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

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.gguf.q4_0.bin",
    temperature=0.3,
    max_tokens=3000,
    top_p=0.73,
    top_k=0,
    n_ctx=2000,
    callback_manager=callback_manager,
    verbose=True, # Verbose is required to pass to the callback manager
)

vex_vectorstore = Chroma(persist_directory=vex_persist_directory, embedding_function=embedding)
vex_retriever = vex_vectorstore.as_retriever(search_kwargs={'k': 3})
vex_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vex_retriever, verbose=True
)

tools = load_tools(["google-serper", "llm-math"], llm=llm)
tools.append(Tool(name="VEX", func=vex_chain.run, description="useful for when you need to answer questions about the VEX documents, which are security advisories in the format RHSA-XXXX:XXXX, where X can be any number."))

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=4,
    return_messages=True,
    output_key="output"
)

class OutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            response = parse_json_markdown(text)
            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                return AgentFinish({"output": action_input}, text)
            else:
                return AgentAction(action, action_input, text)
        except Exception:
            return AgentFinish({"output": text}, text)

    @property
    def _type(self) -> str:
        return "conversational_chat"

parser = OutputParser()

agent_executor = initialize_agent(
    tools,
    llm,
    agent="chat-conversational-react-description",
    # If the agent never returns a ActionFinish then this parameter will
    # determins what should be done, it can either be "force" which just returns
    # a message that the max iterations/timeout has been reached or "generate"
    # where the LLM is given a last chance to generate a response.
    early_stopping_method="generate", 
    return_intermediate_steps=False,
    verbose=True,
    max_iterations=10,
    memory=memory,
    agent_kwargs={"output_parser": parser}
)
#print(agent_executor.agent.llm_chain.prompt)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<>\n", "\n<>\n\n"

sys_msg = B_SYS + """Assistant is designed to assist with a wide range of tasks.

Assistant is able to respond to the User and use tools using JSON strings that contain "action" and "action_input" parameters. Never include ``` in your messages. 

All of Assistant's communication is performed using this JSON format.

Assistant can also use tools by responding to the user with tool use instructions in the same "action" and "action_input" JSON format. Tools available to Assistant are:

- "google-serper": Useful for when you search the web for answers.
  - To use the goggle-serper tool, Assistant should write like so:
    ```json
    {{"action": "google-serper",
      "action_input": "query"}}
    ```

- "VEX": Useful for when you need to answer questions about security advisory VEX documents in the format RHSA-XXXX:XXXX, where X can be any number. Don't use this tool for details about CVEs but instead use 'google-serper' for that.
  - To use the VEX tool, Assistant should write like so:
    ```json
    {{"action": "VEX",
      "action_input": "RHSA-XXXX:XXXX"}}
    ```

Here are some previous conversations between the Assistant and User:

User: Can you show me a summary of the security advisory RHSA-2020:5566?
Assistant:
```json
{{"action": "VEX",
 "action_input": "RHSA-2020:5566"}}
```
Observation: RHSA-2020:5566 is a security advisory related to openssl and has...
Assistant:
```json
{{"action": "Final Answer",
 "action_input": "RHSA-2020:5566 is a security advisory related to openssl and has..."}}
```

User: Could you explain what CVE-2020-14386 is about?
Assistant:
```json
{{"action": "google-serper",
 "action_input": "CVE-2020-14386"}}
```
User: 16.0
Assistant:
```json
{{"action": "Final Answer",
 "action_input": "CVE-2020-14386 is a vulnerability in the Linux kernel that allows a local attacker to cause a denial of service (DoS) or possibly execute arbitrary code on a system."}}
```

Here is the latest conversation between Assistant and User.""" + E_SYS
new_prompt = agent_executor.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)
agent_executor.agent.llm_chain.prompt = new_prompt

instruction = B_INST + " Respond to the following in JSON with 'action' and 'action_input' value(s) " + E_INST
human_msg = instruction + "\nUser: {input}"
#human_msg = "\nUser: {input}"

agent_executor.agent.llm_chain.prompt.messages[2].prompt.template = human_msg

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
