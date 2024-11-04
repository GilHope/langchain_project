import os
from dotenv import load_dotenv
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load OpenAI API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is missing!")

# Verify EPUB file exists
epub_path = 'birth_of_tragedy.epub' 
if not os.path.exists(epub_path):
    raise FileNotFoundError(f"EPUB file not found at path: {epub_path}")

# Function to load and parse EPUB with enhanced metadata
def load_epub(epub_path):
    book = epub.read_epub(epub_path)
    documents = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract the title from the HTML content
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            else:
                title = item.get_name()  # Fallback to the item name
            
            text = soup.get_text()
            text = text.replace('\n', ' ').replace('\r', ' ').strip()
            if text:
                metadata = {'source': title}
                documents.append(Document(page_content=text, metadata=metadata))
    return documents

# Load EPUB file
documents = load_epub(epub_path)

# Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Assign chunk numbers
for i, doc in enumerate(docs):
    doc.metadata['chunk'] = i

# Ensure there are documents to process
if not docs:
    raise ValueError("No documents to process after splitting.")

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Store embeddings in vector store
vector_store = FAISS.from_documents(docs, embeddings)

# Setup Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize OpenAI LLM
llm = ChatOpenAI(openai_api_key=api_key, temperature=0)

# Prompt Template with 'summaries'
prompt_template = """You are an expert assistant. Use only the following context to answer the question. Do not use any outside knowledge, and do not make any assumptions. If the answer is not contained within the context, please say "I don't know."

Context: 
{summaries}

Question: 
{question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["summaries", "question"]
)

# Create QA Chain with sources
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# Define query
query = "What is the difference between the Apollonian and Dionysian?"

# Run QA Chain 
result = qa_chain.invoke({"question": query})

# Print answer and sources with chunk numbers
print("Answer:")
print(result['answer'])
print("\nSources:")
unique_sources = set()
for doc in result['source_documents']:
    source = doc.metadata.get('source', 'Unknown Source')
    chunk = doc.metadata.get('chunk', 'Unknown Chunk')
    source_info = f"{source} (Chunk {chunk})"
    unique_sources.add(source_info)
for source_info in unique_sources:
    print(source_info)
