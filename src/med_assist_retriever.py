from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from loader.file_loader import load_csv_file

load_dotenv()
#setup env varibles
pc_key = os.getenv('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY']=pc_key
google_api_key = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = google_api_key

data =load_csv_file()
textChunks= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100).split_documents(data)
google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db = PineconeVectorStore(
    index_name="med-chat-common",
    embedding=google_embeddings,
    pinecone_api_key=pc_key,
)

vector_data = db.add_documents(documents=textChunks)

if (vector_data):
    print("Successfully added Data")
else:
    print("Retry")

def med_assist_retriver_chroma_db(db_path):
    ins_vectordb_data =Chroma(
        persist_directory=f"../../{db_path}/",
        embedding_function=google_embeddings,
    )
    print(ins_vectordb_data)
    return ins_vectordb_data

def med_assist_retriver_pinecone_db():
    try:
        ins_pinecon_data = db
        return ins_pinecon_data
    except Exception as e:
        logging.error(f"Error creating PineconeVectorStore: {e}")
        return None

