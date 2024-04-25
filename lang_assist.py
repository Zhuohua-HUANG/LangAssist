import os
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import openai_model

class LangAssist(object):
    def __init__(self):
        LLM_model = openai_model.ChatLLM()
        self.llm = LLM_model.get_llm()
        self.embeddings = LLM_model.get_embeddings()

        self.vs_folder_name = "vectorstore"
        self.vs_index_name = "course_info"

        if not os.path.exists(self.vs_folder_name+"/"+self.vs_index_name+".faiss"):
            ### Construct retriever ###
            loader = JSONLoader(
                file_path='./data/all_courses_v2.json',
                jq_schema='.[].text',
                text_content=False
            )
            docs = loader.load()


            vectorstore = FAISS.from_documents(docs, self.embeddings)
            vectorstore.save_local(folder_path=self.vs_folder_name, index_name=self.vs_index_name)

        vectorstore = FAISS.load_local(folder_path=self.vs_folder_name, index_name=self.vs_index_name, embeddings=self.embeddings,
                                       allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.
        
        {context}
        
        Question: {question}
        
        Helpful Answer:"""
        custom_rag_prompt = PromptTemplate.from_template(template)

        self.rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | custom_rag_prompt
                | self.llm
                | StrOutputParser()
        )
    def get_answer(self, question):
        for chunk in self.rag_chain.stream(
                question):
            yield chunk
