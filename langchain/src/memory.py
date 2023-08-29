from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory

from dotenv import load_dotenv, find_dotenv

import os
import datetime

_ = load_dotenv(find_dotenv())
openapi_api_key = os.environ['OPENAI_API_KEY']

# Account for deprecation of LLM model
current_date = datetime.datetime.now().date()
target_date = datetime.date(2024, 6, 12)
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"
print(f'Using model: {llm_model}')

llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=False
)

print(conversation.predict(input="Hi, my name is Daniel"))
print(conversation.predict(input="What is 1+1?"))
print(conversation.predict(input="What is my name?"))

print(f'{memory.buffer=}')
print(f'{memory.load_memory_variables({})=}')


memory.save_context({"input": "I'm 180cm tall"},
                    {"output": "That is pretty normal"})
print(conversation.predict(input="How tall am I?"))

memory = ConversationBufferWindowMemory(k=1)
llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=True
)
print(conversation.predict(input="Hi, my name is Daniel"))
print(conversation.predict(input="What is 1+1?"))
print(conversation.predict(input="What is my name?"))
