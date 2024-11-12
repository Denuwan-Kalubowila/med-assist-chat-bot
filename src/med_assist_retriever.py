""" This module contains the function to retrieve the PineconeVectorStore instance"""
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from absl import logging

load_dotenv()

pc_key = os.getenv('PINECONE_API_KEY')
if not pc_key:
    logging.error("GOOGLE_API_KEY environment variable is not set.")
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
os.environ['PINECONE_API_KEY']=pc_key

google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key:
    logging.error("GOOGLE_API_KEY environment variable is not set.")
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
os.environ['GOOGLE_API_KEY'] = google_api_key

google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db = PineconeVectorStore(
    index_name="medichat",
    embedding=google_embeddings,
    pinecone_api_key=pc_key,
)

def med_assist_retriver_pinecone_db():
    """
    Retrieves the PineconeVectorStore instance for the medical assistant.

    Returns:
        PineconeVectorStore: The instance of the PineconeVectorStore for the medical assistant.
        None: If there was an error creating the PineconeVectorStore.
    """
    try:
        ins_pinecon_data = db
        return ins_pinecon_data
    except Exception as e:
        logging.error(f"Error creating PineconeVectorStore: {e}")
        return None