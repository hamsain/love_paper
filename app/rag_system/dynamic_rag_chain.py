import os

from langchain_core.runnables import RunnablePassthrough, RunnableGenerator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from dotenv import load_dotenv
from typing import AsyncIterator

# Load environment variables
load_dotenv()

# === Configuration ===
groq_api_key = os.environ['GROQ_API_KEY']
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# === Embedding and Chat Model ===
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

groq_chat = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="deepseek-r1-distill-llama-70b",
    streaming=True  # Enable streaming here
)

# === Prompt Template ===
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provide answers to questions by using fact-based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know — don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.
Please answer in the same language as the question.

Assistant:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# === Utils ===
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# === Chain Builder ===
def build_rag_chain(index_name: str) -> RunnableGenerator:
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=embedding)
    retriever = vectorstore.as_retriever()

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | groq_chat
        | StrOutputParser()
    )



# Optional: add interactive playground at `/langserve`
# add_routes(app, build_rag_chain("pertanian-384"), path="/langserve")
