from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from constants import prompt_template_str

def get_response(question, context, gpt_api_key, temp=0.7, model="gpt-3.5-turbo"):
    """Generates a response from Gemini using the provided context."""
    chat = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=gpt_api_key,
        model_name=model,  # Or any other model supported by OpenRouter
        temperature=temp
    )
    prompt_template = PromptTemplate.from_template(template=prompt_template_str)
    prompt = prompt_template.format(context=context, question=question)

    response = chat.invoke(prompt)
    return response.content
