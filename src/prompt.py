from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate

def custom_prompt_template():
    template_str = """You are a medical assistant with expert knowledge in common medical problems faced by patients in day-to-day life. 
Your task is to provide accurate, reliable, and responsible answers to solve problems faced by patients. 
Use the given context to ensure accuracy in your answers. 
If you are unsure about an answer, state that you don't know. 
Your answers should be in an ordered list, short and concise.
{context}
"""

    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],
            template=template_str
        )
    )

    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template="{question}"
        )
    )

    custom_template = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            MessagesPlaceholder("chat_history"),
            human_prompt,
        ]
    )

    return custom_template

def history_chat_template():
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
that can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],
            template=contextualize_q_system_prompt
        )
    )

    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question", "chat_history"],  # Include both "question" and "chat_history"
            template="{question}"
        )
    )

    custom_history_template = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            MessagesPlaceholder("chat_history"),
            human_prompt,
        ]
    )

    return custom_history_template

