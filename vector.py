from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# folder to store database
db_location = "./chroma_langchain_db"

# only if the database does not exist, we will add documents
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # create document for each review
        document = Document(
            page_content=row['Title'] + " " + row["Review"],
            metadata={"rating": row['Rating'], "date": row["Date"]},
            id = str(i)
        )

        ids.append(str(i))
        documents.append(document)

# create the vector store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location, # directory to store the database
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents, ids=ids)

# retrieving data in the vector store
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5} # number of documents to retrieve
)