from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
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


chat = ChatOpenAI(temperature=0.0, model=llm_model)

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

customer_style = """American English \
in a calm and respectful tone
"""

customer_email = """
Jag var väldigt arg att att min säng gick sönder \
första gången jag la mig på den. \
Jag har förvisso gått upp i vikt men inte så mycket.
"""

customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

print(customer_messages)

customer_response = chat(customer_messages)
print(customer_response)
