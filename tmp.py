from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import json
import datasets
from openai_model import ChatLLM

pd.set_option("display.max_colwidth", None)

# TODO: should be changed to the knowledge base
ds = datasets.load_dataset("json", data_files="data/all_courses.json")
print(ds)


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

langchain_docs = [LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)]
print(langchain_docs)