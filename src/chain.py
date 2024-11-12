""" This module defines the QA chain for the medical assistant chatbot. """
import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from absl import logging
from src.med_assist_retriever import med_assist_retriver_pinecone_db
from src.prompt import custom_prompt_template_agent,custom_prompt_template
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain.agents import create_react_agent,AgentExecutor
from langchain import hub


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
med_template = custom_prompt_template_agent()
med_context_template = custom_prompt_template()

# Initialize the retriever database
retriever_db = med_assist_retriver_pinecone_db()

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=4,
    return_messages=True 
)

qa =(
        {
            "context": retriever_db.as_retriever(),
            "question": RunnablePassthrough(),
        } 
        | med_context_template
        | model 
        | output_parser
)

tools =[
    Tool(
        name='Medical Advisor',
        func= qa.invoke,
        description=(
            "This tool provides medical advice to patients based on their symptoms."
        )
    )
]

agent = create_react_agent(
    tools=tools,
    llm=model,
    prompt=med_template,
)

agent_executor = AgentExecutor(
    agent=agent,
    memory=conversational_memory,
    tools=tools,
    verbose=True,
    max_iterations=30,
    max_execution_time=100,
    handle_parsing_errors=True,
)

def handle_user_input(user_input: str):
    """Process user input using the medical assistant chain."""
    try:
        # Invoke the agent executor with user input
        response = agent_executor.invoke({"input": user_input})
        return response['output'],response['chat_history']
    except Exception as e:
        logging.error(f"Error processing input: {e}")
        return "Sorry, something went wrong while processing your request."


