import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate



### Load OpenAI API Key ###
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is missing!")

### Load and Extract Text from PDF ###
pdf_path = 'birth_of_tragedy.pdf'

loader = PyPDFLoader(pdf_path)
documents = loader.load()

### Split Text to Chunks ###
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

docs = text_splitter.split_documents(documents)

### Create embeddings for Big Chunkus ###
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

### Store embeddings in vector store ###
vector_store = FAISS.from_documents(docs, embeddings)

### Setup Retriever ###
retriever = vector_store.as_retriever()

### Initialize OpenAI ###
llm = OpenAI(openai_api_key=api_key, temperature=0)

### Prompt Template ###
prompt_template = """You are an expert assistant. Use only the following context to answer the question. Do not use any outside knowledge. If the answer is not contained within the context, please say "I don't know".

Context: 
{context}

Question: 
{question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

### Create QA Chain ###
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    input_key='question'
)

query = "What is the difference between the Apollonian and Dionysian?"

# query = "What is the capital of france?"

### Run QA Chain and get answer ###

# answer = qa_chain.invoke(query)
# print(answer)

answer = qa_chain.invoke({"question": query})
print(answer['result'])