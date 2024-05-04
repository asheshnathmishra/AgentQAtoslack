"""
    This file builts a Langchain Agent that talks with a PDF file using Langchain Agent.
"""

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI

import os
import chromadb

os.environ["OPENAI_API_KEY"] = "key"

# Load the PDF file
loader = PyPDFLoader(r"C:\Users\ashes\OneDrive\Code\Zania.AI assignment\handbook.pdf")
pages = loader.load_and_split()

# Split the PDF content into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(pages)

# Create embeddings for the PDF chunks
embeddings = OpenAIEmbeddings()  # Replace with your actual API key
vectorstore = Chroma.from_documents(texts, embeddings)

# Initialize the chat model
chat = ChatOpenAI(temperature=0.7)  # Replace with your actual API key

# Create the conversational chain
qa = ConversationalRetrievalChain.from_llm(chat, vectorstore.as_retriever())

tools = [
    Tool(
        name="Open AI Playground",
        func=lambda q: q,
        description="Useful for exploring and analyzing text. Input should be a query for exploring or analyzing the given text."
    )
]
agent = initialize_agent(tools, chat, agent="conversational-react-description", verbose=True)

# Chat with the PDF
print("Ask your questions about the PDF:")
chat_history = []  # Initialize an empty list to store the chat history

while True:
    query = input("> ")
    if query.lower() == "exit":
        break

    # Use the conversational chain or agent to get the response
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    print("user: ",query)
    print(f"Answer: {result['answer']}")