from dotenv import load_dotenv
import os
import pinecone
from langchain_core.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key
pc_key = os.getenv('PINECONE_API_KEY')
pc_env = os.getenv('PINECONE_ENVIRONMENT')

pinecone.init(api_key=pc_key, environment=pc_env)


PATH_CSV = "../data/train.csv"

# Load the CSV file
loader = CSVLoader(PATH_CSV)
data = loader.load()
text_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50).split_documents(data)

print(text_chunks, end="\n")

def create_pinecone_vectorstore(index_name):
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = Pinecone(
            index=index_name,
            embedding=embeddings,
            text_key=pc_key,
        )
        return vectorstore
    except Exception as e:
        logging.error(f"Error creating Pinecone vector store: {e}")
        return None

# Creating the Pinecone vector store
index_name = "med-chat-data"
vectorstore = create_pinecone_vectorstore(index_name)

# Check if the vector store was created successfully
if vectorstore:
    for chunk in text_chunks:
        vectorstore.add_texts([chunk])
    print(f"Data successfully added to the {index_name} index in Pinecone.")
else:
    print("Failed to create Pinecone vector store.")


