from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate,HumanMessagePromptTemplate,PromptTemplate

def custom_prompt_template ():
    template_str = """You are a medical assistant with expert knowledge in common medical problems faced by patients in day-to-day life. 
                    Your task is to provide accurate, reliable, and responsible answers to solve problems faced by patients. 
                    Use the given context to ensure accuracy in your answers. 
                    Your answers should be in an ordered list, short and concise.
                    If you don't know the answer say "I don't know"
                    {context}
            """

    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],template=template_str
        )
    )

    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],template="{question}"
        )
    )

    messages = [system_prompt,human_prompt]

    custom_template = ChatPromptTemplate(
        input_variables=["context","question"],
        messages=messages,
    )

    return custom_template



