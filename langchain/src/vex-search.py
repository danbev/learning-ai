from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
import numpy as np
import os

load_dotenv()

loader = JSONLoader(file_path='./src/vex-stripped.json', jq_schema='.document', text_content=False)

docs = loader.load()
#print(f'Pages: {len(docs)}, type: {type(docs[0])})')
#print(f'{docs[0].metadata}')

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=1,
    length_function=len  # function used to measure chunk size
)

splits = r_splitter.split_documents(docs)
print(f'splits len: {len(splits)}, type: {type(splits[0])}')


embedding = OpenAIEmbeddings()
persist_directory = 'chroma/sds2/'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cos_sim = dot_product / (norm_vec1 * norm_vec2)
    return cos_sim

template = """I will provide you pieces of [Context] to answer the [Question]. 
If you don't know the answer based on [Context] just say that you don't know, don't try to make up an answer. 
Include a bulleted list of CVE's and references at the end of your answer.

[Context]: {context} 
[Question]: {question} 
Helpful Answer:"""

prompt_template = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=False)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(), # will use vectordb to retrieve documentssrelated to the query
    chain_type="stuff", # "stuff" as in stuff the documents retreived into the template.
    verbose=False,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template, "verbose": False,}
)

question = "Summarize RHSA-2020:5566 using a short sentence, including a list of CVE's and references."
result = qa_chain({"query": question})

print('Answer:') 
print(result["result"])

print('\n\nSource documents:') 
for doc in result['source_documents']:
    print(f'{doc.metadata["source"]}')
