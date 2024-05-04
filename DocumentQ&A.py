import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
import requests

def load_and_chunk_pdf(pdf_path, chunk_size=1000, chunk_overlap=300):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(pages)

    return texts

def initialize_openai_and_agent(openai_api_key):
    chat = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)

    tools = [
        Tool(
            name="Open AI Playground",
            func=lambda q: q,
            description="Useful for exploring and analyzing text. Input should be a query for exploring or analyzing the given text."
        )
    ]
    agent = initialize_agent(tools, chat, agent="conversational-react-description", verbose=True)

    return chat, agent

def main():
    openai_api_key = os.environ.get("key")
    pdf_path = r"C:\Users\ashes\OneDrive\Code\Zania.AI assignment\handbook.pdf"
    slack_webhook_url = "https://hooks.slack.com/services/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"

    texts = load_and_chunk_pdf(pdf_path)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(texts, embeddings)

    chat, agent = initialize_openai_and_agent(openai_api_key)
    qa = ConversationalRetrievalChain.from_llm(chat, vectorstore.as_retriever())

    print("Ask your questions about the PDF:")
    chat_history = []

    while True:
        query = input("> ")
        if query.lower() == "exit":
            break

        result = qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, result["answer"]))
        print(f"Question: {query}")
        print(f"Answer: {result['answer']}")

        slack_message = {
            "text": f"Question: {query}\nAnswer: {result['answer']}"
        }
        response = requests.post(slack_webhook_url, json=slack_message)
        if response.status_code != 200:
            print(f"Failed to post message to Slack: {response.text}")

if __name__ == "__main__":
    main()