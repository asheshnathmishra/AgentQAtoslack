# Document Q&A Bot with Slack Integration
This project implements a Question & Answer (Q&A) bot that processes a document (PDF file) and interacts with users to answer questions based on the document's content. The bot utilizes LangChain's agent and conversation history is pushed into a Slack channel for collaboration.

**Features**
Extracts text from a PDF document and splits it into manageable chunks for processing.
Utilizes OpenAI's embeddings to represent document chunks in a vector space.
Implements a Conversational Retrieval Chain for answering user queries based on the document content.
Integrates with Slack using webhook URLs to push conversation history in real-time.
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
