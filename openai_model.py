import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

os.environ["OPENAI_API_BASE"] = "https://cfcus02.opapi.win/v1"

with open('APIKEY', 'r') as file:
    openai_api_key = file.read()


class ChatLLM:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4-turbo-preview",
            temperature=0,
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
        )

    def get_llm(self):
        return self.llm

    def get_embeddings(self):
        return self.embeddings
    
    def call_llm(self, prompt: str) -> str:
        answer = self.llm.invoke(
            prompt
        ).content
        return answer
