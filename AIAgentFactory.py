import streamlit as st
import google.generativeai as genai

from constants import GEMINI_AGENT, GPT4_AGENT, GPT4_AGENT_ROUTE_API
from ai_agents.GeminiAIAgent import get_response as get_gemini_response
from ai_agents.ChatGPTRouteAIAgent import get_response as get_gpt_route_response
from ai_agents.ChatGPTAIAgent import get_response as get_gpt_response

class GetAIAgentFactory:
    def __init__(self):
        self.agentType = int(st.secrets.get("AGENT_TYPE", "1"))
        self.temp = float(st.secrets.get("TEMPERATURE", "0.0"))

    def get_response(self, question, context):
        if self.agentType == GEMINI_AGENT:
            gemni_modal = st.secrets.get("GEMINI_MODAL", "gemini-2.5-flash-lite")
            google_api_key = st.secrets.get("GOOGLE_API_KEY")
            
            print(f"LLM {gemni_modal} selected")

            if not google_api_key:
                st.error("API keys not found. Please add them to your Streamlit secrets.")
                st.stop()

            genai.configure(api_key=google_api_key)

            return get_gemini_response(question, context, self.temp, gemni_modal)
        elif self.agentType == GPT4_AGENT:
            gpt_modal = st.secrets.get("GPT_MODAL", "gpt-3.5-turbo")
            gpt_api_key = st.secrets.get("OPENAI_API_KEY")

            print(f"LLM {gpt_modal} selected")

            if not gpt_api_key:
                st.error("API keys not found. Please add them to your Streamlit secrets.")
                st.stop()

            return get_gpt_response(question, context, gpt_api_key, self.temp, gpt_modal)
        elif self.agentType == GPT4_AGENT_ROUTE_API:
            gpt_modal = st.secrets.get("GPT_ROUTE_AI_MODAL", "gpt-4o-mini")
            gpt_api_key = st.secrets.get("OPENROUTER_API_KEY")

            print(f"LLM {gpt_modal} selected")

            if not gpt_api_key:
                st.error("API keys not found. Please add them to your Streamlit secrets.")
                st.stop()

            return get_gpt_route_response(question, context, gpt_api_key, self.temp, gpt_modal)
        return None
