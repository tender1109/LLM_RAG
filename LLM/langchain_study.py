from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following into {language}:"


prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

result = prompt_template.invoke({"language": "italian", "text": "hi"})

print(result)






