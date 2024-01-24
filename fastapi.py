# -*- coding: utf-8 -*-
"""fastapi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JpCV53oTsjUT_JEQtT5fPivXkV_Y7Wg7
"""

!pip install -q cassio datasets langchain openai tiktoken

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

app = FastAPI()

# Initialize your database and OpenAI components here
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:LhaZBxXUFSRnsBgjuTxifUPY:085e6f320623eae8aff635be6fdf6e0854ad9c961f867d463add66b5ca66c3f2" # enter the "AstraCS:..." string found in in your Token JSON file
ASTRA_DB_ID = "07a929a2-9a48-4d66-80f0-7c28c582d47c" # enter your Database ID

OPENAI_API_KEY = "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" # enter your OpenAI key
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

class Query(BaseModel):
    text: str

@app.post("/add-texts/")
async def add_texts(file: UploadFile = File(...)):
    try:
        pdfreader = PdfReader(await file.read())
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

        # Text processing and storing in database
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)
        astra_vector_store.add_texts(texts[:50])
        return {"message": f"Inserted {len(texts[:50])} chunks of text."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query(query: Query):
    try:
        # Querying logic
        answer = astra_vector_index.query(query.text, llm=llm).strip()

        # Getting the first documents by relevance
        relevance_docs = [
            {"score": score, "content": doc.page_content[:84]}
            for doc, score in astra_vector_store.similarity_search_with_score(query.text, k=4)
        ]

        return {
            "answer": answer,
            "relevant_documents": relevance_docs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# You can add more endpoints as needed

