from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import JSONLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

def load_cve_docs():
    cve_loader = DirectoryLoader('src', glob="cve*", loader_cls=JSONLoader,
                                 loader_kwargs = {'jq_schema': '.', 'text_content': False})
    cve_docs = cve_loader.load()
    cve_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len  # function used to measure chunk size
    )
    cve_splits = cve_splitter.split_documents(cve_docs)
    print(f'cve_splits len: {len(cve_splits)}, type: {type(cve_splits[0])}')
    cve_vectorstore = Chroma.from_documents(
        documents=cve_splits,
        embedding=embedding,
        persist_directory=cve_persist_directory
    )
    cve_vectorstore.persist()

# This only need to be run once to create the vector store, or after more
# documents are to be added to the store.
#load_vex_docs()
#load_cve_docs()

import sys

if len(sys.argv) > 1:
    advisory = sys.argv[1]
else:
    advisory = "RHSA-2020:5566"

vex_vectorstore = Chroma(persist_directory=vex_persist_directory, embedding_function=embedding)
vex_retriever = vex_vectorstore.as_retriever(search_kwargs={'k': 3})

cve_vectorstore = Chroma(persist_directory=cve_persist_directory, embedding_function=embedding)
cve_retriever = cve_vectorstore.as_retriever(search_kwargs={'k': 6})

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vex_retriever, memory=memory)
query = f'Show a short summary of {advisory}, including the cve.'
print(f'Query: {query}')
result = qa({"question": query})
print(f'Result: {result["answer"]}')

query = "Which CVEs were mentioned"
print(f'Query: {query}')
result = qa({"question": query, "chat_history": result["answer"]})
print(f'Result: {result["answer"]}')

from langchain import PromptTemplate
prompt_template = PromptTemplate.from_template(
        "Show me a detailed description of {cve}."
)
query = prompt_template.format(cve=result["answer"])
print(f'Query: {query}')

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), cve_retriever, memory=memory)
result = qa(inputs={"question": query})
print(f'Result: {result["answer"]}')

