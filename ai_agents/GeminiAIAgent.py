import google.generativeai as genai
from langchain.prompts import PromptTemplate
from constants import prompt_template_str

def get_response(question, context, temp, modal='gemini-2.5-flash-lite'):
    """Generates a response from Gemini using the provided context."""
    chat = genai.GenerativeModel(modal, generation_config={'temperature': temp})
    prompt_template = PromptTemplate.from_template(template=prompt_template_str)
    prompt = prompt_template.format(context=context, question=question)
    
    response = chat.generate_content(prompt)
    return response.text
