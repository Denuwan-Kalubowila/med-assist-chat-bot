from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI,HarmBlockThreshold, HarmCategory,GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from med_assist_retriever import med_assist_retriver_pinecone_db,med_assist_retriver_chroma_db
from prompt import custom_prompt_template

load_dotenv()

# openai_api_key = os.getenv('OPENAI_API_KEY')
# os.environ['OPENAI_API_KEY'] = openai_api_key
google_api_key = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = google_api_key

model = GoogleGenerativeAI(
    model='gemini-pro',
    temperature=0.6,
    google_api_key=google_api_key,
    verbose=True,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

output_parser = StrOutputParser()
med_template = custom_prompt_template()

path_chroma = "chroma_data"
retriever_db =med_assist_retriver_chroma_db(path_chroma)

def med_QA(ques):
    qa_chain=(
        {
            "context": retriever_db.as_retriever(),
            "question": RunnablePassthrough(),
        } 
        | med_template
        | model 
        | output_parser
    )
    return qa_chain.invoke(input=ques)


