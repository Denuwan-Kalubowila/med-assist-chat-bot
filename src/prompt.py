""" this module contains the custom prompt template for the chatbot. """
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate,HumanMessagePromptTemplate,PromptTemplate
def custom_prompt_template_agent ():
    """ 
        Generates a custom prompt template for a medical chatbot.
        Returns:
        custom_template (ChatPromptTemplate): The custom prompt template for the chatbot.
    """

    template_str = """You are a medical assistant with expert knowledge in common medical problems faced by patients in day-to-day life.
                    Your task is to provide accurate, reliable, and responsible answers to solve problems faced by patients.
                    Use the given context to ensure accuracy in your answers. Your responses should focus on humanistic care and empathy. When giving information, present it in an ordered list and keep it short and concise. 
                    If someone asks you a non-medical question, simply say, "I don't know."
                    TOOLS:
                    Assistant has access to the following tools:
                    {tools}
                    To use a tool, please use the following format:
                    Thought: Do I need to use a tool? Yes  
                    Action: the action to take, should be one of [{tool_names}]  
                    Action Input: the input to the action  
                    Observation: the result of the action  

                    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

                    Thought: Do I need to use a tool? No  
                    Final Answer: [your response here]  

                    Begin!
                    Previous conversation history:
                    {chat_history}
                    New input: {input}
                    {agent_scratchpad}


            """
    prompt = ChatPromptTemplate.from_template(template_str)
    return prompt


def custom_prompt_template ():
    
    template_str = """You are a medical assistant with expert knowledge in common medical problems faced by patients in day-to-day life. 
                    Your task is to provide accurate, reliable, and responsible answers to solve problems faced by patients. 
                    Use the given context to ensure accuracy in your answers. 
                    Your answers should be in an ordered list, short and concise.
                    If you don't know the answer say "I don't know". Always your answers focus on humanistic care and empathy.
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



