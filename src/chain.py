from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import HarmBlockThreshold, HarmCategory, ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from med_assist_retriever import med_assist_retriver_pinecone_db, med_assist_retriver_chroma_db
from prompt import custom_prompt_template

load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = google_api_key

model = ChatVertexAI(
    model='gemini-1.5-flash',
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
    
    # Invoke the QA chain with the input question
    return qa_chain.invoke(input=ques)


