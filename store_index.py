from langchain.document_loaders import DirectoryLoader
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
 

#Initializing the Pinecone
# Pinecone(api_key=PINECONE_API_KEY,
#               environment=PINECONE_API_ENV)

#init Pinecone
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV )

index_name="medical-chatbot"

index = pinecone_client.Index(index_name)

#Creating Embeddings for Each of The Text Chunks & storing
# docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

metadata = {}
for t in text_chunks:
    embedding = embeddings.embed_query(t.page_content)
    metadata["text"] = t.page_content
    index.upsert(vectors=[
        {
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": metadata
        }
    ])