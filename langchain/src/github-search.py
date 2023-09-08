from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import GitLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import numpy as np

import os

load_dotenv()

loader = GitLoader("./checkout_dir", "https://github.com/danbev/learning-iot.git", "master")
docs = loader.load()
#print(f'Pages: {len(docs)}, type: {type(docs[0])})')
#print(f'{docs[0].metadata}')


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len  # function used to measure chunk size
)

splits = r_splitter.split_documents(docs)
#print(f'splits len: {len(splits)}, type: {type(splits[0])}')


embedding = OpenAIEmbeddings()
persist_directory = 'chroma/sds/'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

question = "What frequencies does LoRa use"
docs = vectordb.similarity_search(question, k=5)
for doc in docs:
    print(f'{doc.metadata["file_name"]}, {doc.metadata["source"]}')


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cos_sim = dot_product / (norm_vec1 * norm_vec2)
    return cos_sim

q_emb = embedding.embed_query(question)
q_vec = np.array(q_emb)

for d in docs:
    emb = embedding.embed_query(d.page_content)
    vec = np.array(emb)
    cosine = cosine_similarity(q_vec, vec)
    print(f'{cosine=}, {d.metadata["file_name"]}')

template = """I will provide you pieces of [Context] to answer the [Question]. 
If you don't know the answer based on [Context] just say that you don't know, don't try to make up an answer. 
Translate the answer into Swedish if you can. If you cannot translate the answer,
just say that you cannot translate it show the aswer in the English.

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

question = "Explain what LoRa is in the context of IoT"
result = qa_chain({"query": question})

print('Answer:') 
print(result["result"])

print('\n\nsource_documents:') 
for doc in result['source_documents']:
    print(f'{doc.metadata["source"]}')
