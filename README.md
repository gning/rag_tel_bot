# Telegram PDF RAG Bot

A Telegram bot that implements a Retrieval-Augmented Generation (RAG) system for PDF documents. Users can upload PDF files to the bot, and then ask questions about the content. The bot uses a vector database to store document chunks and perform semantic search to find relevant information.

## Features

- **PDF Processing**: Extract text from uploaded PDF files and split it into chunks
- **Vector Database**: Store document chunks with embeddings in Pinecone
- **Semantic Search**: Find relevant information based on user queries
- **LLM Integration**: Generate responses using Google's Gemini model
- **Source Attribution**: Provide source filenames for responses

## Technologies Used

- Python 3.9+
- python-telegram-bot
- Pinecone (vector database)
- Google Generative AI (gemini-2.0-pro-exp-02-05 for LLM and models/embedding-001 for embeddings)
- PyPDF2 (PDF processing)
- LangChain (text processing and vectorstore integration)

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/rag_tel_bot.git
cd rag_tel_bot
```

2. **Create a virtual environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Create a Telegram bot**

- Talk to [@BotFather](https://t.me/BotFather) on Telegram
- Use the `/newbot` command to create a new bot
- Copy the bot token provided by BotFather

4. **Set up Pinecone**

- Create an account at [Pinecone](https://www.pinecone.io/)
- Create a new serverless index with dimension 768 (for Gemini text-embedding-005) and cosine similarity metric
- Copy your API key and index name

5. **Set up Google Gemini API**

- Create/use a Google Cloud account and enable the Gemini API
- Generate an API key for Gemini

6. **Environment Variables**

Copy the `.env.example` file to `.env` and fill in your actual API keys and configuration:

```bash
cp .env.example .env
# Edit the .env file with your credentials
```

## Running the Bot

Start the bot with the following command:

```bash
python bot.py
```

## Usage

1. **Start a conversation with your bot** on Telegram
2. Send the bot a PDF file to process
3. Ask questions about the content of the PDF
4. The bot will respond with information from the PDF along with the source filename

## Example Commands

- `/start` - Start the bot
- `/help` - Show help message
- Send a PDF file to process it
- Ask a question by just typing it

## Notes

- The bot uses a default chunk size of 256 words, which can be adjusted in the `.env` file
- The bot retrieves the top 5 most relevant results for each query by default
- For best results, use clear and specific questions that relate to the content in your PDF documents

## License

This project is licensed under the terms of the MIT license.
