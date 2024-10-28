import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA



load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is missing!")

# response = llm.invoke("Hello, World!")
# print(response)

# response = llm.invoke("Hello, World! How are you today?", max_tokens=100, temperature=0.7, top_p=0.9)
# print(response)

# Load and Extract Text from PDF
pdf_path = 'birth_of_tragedy.pdf'

loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split Text to Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

docs = text_splitter.split_documents(documents)

# Create embeddings for Big Chunkus
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Store embeddings in vector store
vector_store = FAISS.from_documents(docs, embeddings)

# Setup Retriever
retriever = vector_store.as_retriever()

# Initialize OpenAI
llm = OpenAI(openai_api_key=api_key, temperature=0)

# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "What is the distinction according to Nietzsche between the Apollonian and the Dionysian?"

# answer = qa_chain.invoke(query)
# print(answer)

answer = qa_chain.invoke({"query": query})
print(answer['result'])