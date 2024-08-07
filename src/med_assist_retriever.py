from dotenv import load_dotenv
import os
from langchain_community.document_loaders import DirectoryLoader,CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

load_dotenv()

pc_key = os.getenv('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY']=pc_key
google_api_key = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = google_api_key

# Check if the OpenAI API key is available
PATH_CSV = "data"
PATH_CHROMA = "chroma_data"

# Load the CSV file
loader = DirectoryLoader(
    "../data", 
    show_progress=True,
    loader_cls=CSVLoader
)
data = loader.load()
textChunks= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(data)
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

def med_assist_retriver_pinecone_db():
    try:
        ins_pinecon_data = db
        return ins_pinecon_data
    except Exception as e:
        logging.error(f"Error creating PineconeVectorStore: {e}")
        return None