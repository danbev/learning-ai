from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import pandas as pd
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

df = pd.read_csv('src/Data.csv')
print(df.head())

llm = ChatOpenAI(temperature=0.9, model=llm_model)

prompt = ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")
# This "chain" is the combination of the llm and the prompt.
chain = LLMChain(llm=llm, prompt=prompt)

product = "Large pillow"
result = chain.run(product)
print(result)


from langchain.chains import SimpleSequentialChain

first_prompt = prompt
chain_one = LLMChain(llm=llm, prompt=first_prompt)

second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )
result = overall_simple_chain.run(product)
print(result)
