from dotenv import load_dotenv
import os
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from absl import logging
from src.med_assist_retriever import med_assist_retriver_pinecone_db
from src.prompt import custom_prompt_template

load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key:
    logging.error("GOOGLE_API_KEY environment variable is not set.")
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
os.environ['GOOGLE_API_KEY'] = google_api_key

model = ChatVertexAI(
    model='gemini-1.5-flash',
    project='medassist-419918',
)

output_parser = StrOutputParser()
med_template = custom_prompt_template()

# Initialize the retriever database
retriever_db = med_assist_retriver_pinecone_db()

def med_QA(ques):
    # Define the QA chain
    qa_chain = (
        {
            "context": retriever_db.as_retriever(),
            "question": RunnablePassthrough(),
        } 
        | med_template
        | model 
        | output_parser
    )
    
    return qa_chain.invoke(input=ques)


