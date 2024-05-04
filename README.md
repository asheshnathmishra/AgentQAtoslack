# Document Q&A Bot with Slack Integration
This project implements a Question & Answer (Q&A) bot that processes a document (PDF file) and interacts with users to answer questions based on the document's content. The bot utilizes LangChain's agent and conversation history is pushed into a Slack channel for collaboration.

**Features**

Extracts text from a PDF document and splits it into manageable chunks for processing.
Utilizes OpenAI's embeddings to represent document chunks in a vector space.
Implements a Conversational Retrieval Chain for answering user queries based on the document content.
Integrates with Slack using webhook URLs to push conversation history in real-time.

**Methods Used**

**1. load_and_chunk_pdf(pdf_path, chunk_size=1000, chunk_overlap=300)**
Description: This function loads a PDF document from the specified path and splits it into smaller chunks of text to facilitate processing. It utilizes LangChain's PyPDFLoader and CharacterTextSplitter for PDF loading and text splitting, respectively.

**2. initialize_openai_and_agent(openai_api_key)**
Description: This function initializes the OpenAI chat model and sets up an agent for interaction. It configures the chat model with the provided OpenAI API key and defines a tool for the agent to interact with. The agent is initialized with LangChain's initialize_agent function, which sets up the conversational agent with the specified tools and chat model.

**3. main()**
Description: The main function of the script orchestrates the entire workflow of the Q&A bot. It loads the PDF document, initializes the OpenAI chat model and agent, processes user queries, retrieves answers from the document, and pushes conversation history to a Slack channel using a webhook URL.

**Dependencies**

Python 3.x
langchain (install via pip)
requests (install via pip)
chromadb
OpenAI

**Configuration**

Modify DocumentQ&A.py to change the PDF file path (pdf_path), set OpenAI key and adjust parameters such as temperature, chunk size and chunk overlap for the OpenAI chat model.
Customize Slack integration by modifying the Slack webhook URL (slack_webhook_url).

**License**

This project is licensed under the MIT License - see the LICENSE file for details.

**Acknowledgments**

This project utilizes LangChain's tools for document processing and conversational AI.
Special thanks to the developers of LangChain and OpenAI for their contributions to natural language processing.
