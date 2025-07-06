from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ['PINECONE_API_KEY']
def do_ingest(uploadedDocName, index_name):
    # Load PDF
    loaders = [
        PyPDFLoader(f"uploads/{uploadedDocName}")
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=250)
    all_splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pc = Pinecone(api_key=api_key)

    index_name = index_name
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384, # Replace with your model dimensions
            metric="cosine", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )

    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    result = vector_store.add_documents(all_splits)
    return result

def remove_extension(filename: str) -> str:
   
    base_name, _ = os.path.splitext(filename)
    return base_name