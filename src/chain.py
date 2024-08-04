from dotenv import load_dotenv
import os
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from med_assist_retriever import med_assist_retriver_pinecone_db, med_assist_retriver_chroma_db
from prompt import custom_prompt_template, history_chat_template

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import secrets
import string

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

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def generate_session_id(length=5):
    characters = string.ascii_uppercase + string.digits
    session_id = ''.join(secrets.choice(characters) for _ in range(length))
    return session_id

def med_QA(ques, session_id=None):
    if session_id is None:
        session_id = generate_session_id()

    # Define the QA chain
    qa_chain = create_stuff_documents_chain(
        llm=model,
        prompt=med_template
    )

    history_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever_db.as_retriever(),
        prompt=history_chat_template(),
    )

    rag_chain = create_retrieval_chain(history_retriever, qa_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    result = conversational_rag_chain.invoke(
        input={"question": ques, "chat_history": get_session_history(session_id).messages},
        config=session_id
    )

    # Store the updated chat history
    store[session_id] = get_session_history(session_id)

    return result, session_id
