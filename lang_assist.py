
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
import openai_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


LLM_model = openai_model.ChatLLM()
llm = LLM_model.get_llm()

### Construct retriever ###
# loader = JSONLoader(
#     file_path='./data/all_courses_v2.json',
#     jq_schema='.[].text',
#     text_content=False
# )
# docs = loader.load()
#
embeddings = LLM_model.get_embeddings()
#
# vectorstore = FAISS.from_documents(docs, embeddings)
# vectorstore.save_local(folder_path='vectorstore', index_name="course_info")
vectorstore = FAISS.load_local(folder_path='vectorstore', index_name="course_info", embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)
for chunk in rag_chain.stream("I am a prospective student and I want to learn a programme related to computer science. Can you compare the courses and differences between information technology and big data technology?"):
    print(chunk, end="", flush=True)