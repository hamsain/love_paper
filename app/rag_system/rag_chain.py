import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Get Groq API key
groq_api_key = os.environ['GROQ_API_KEY']
model = 'deepseek-r1-distill-llama-70b'
# Initialize Groq Langchain chat object and conversation
groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
)

shared_index = ''

# Define the prompt template for generating AI responses
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.
Please answer with the same language as the question.

Assistant:"""

# Create a PromptTemplate instance with the defined template and input variables
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )

from pinecone import Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

from langchain_pinecone import PineconeVectorStore
index_name = "pertanian-384"
# Instantiate the index
index = pc.Index(index_name)
vectorstore = PineconeVectorStore(index=index, embedding=embedding)
retriever = vectorstore.as_retriever()

# Define a function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG (Retrieval-Augmented Generation) chain for AI response generation
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | groq_chat
    | StrOutputParser()
)

# response = rag_chain.invoke("Siapa yang dikerubungi kerumunan supir?")
# print(response)