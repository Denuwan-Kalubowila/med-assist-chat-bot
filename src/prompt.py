from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate,HumanMessagePromptTemplate,PromptTemplate

def custom_prompt_template ():
    template_str = """Your job is to provide correct and responsibele answers of patient's common health problems.
    Use the following context to answer questions. canve Be as detailed
    as possible, but don't make up any information that's not
    from the context. If you don't know an answer, say you don't know.
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



