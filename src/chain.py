from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from med_assist_retriever import med_assist_retriver_pinecone_db,med_assist_retriver_chroma_db
from prompt import custom_prompt_template

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key

model = ChatOpenAI(
    model='gpt-4',
    temperature=0.6
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

ans = med_QA(" What are the treatments for pneumonia")
print(ans)

