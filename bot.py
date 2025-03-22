#!/usr/bin/env python
import os
import logging
from typing import List, Dict, Any, Optional
import io
import tempfile

# Import required libraries
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# PDF processing
from PyPDF2 import PdfReader

# Vector database
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone as LCPinecone

# Gemini for both embeddings and LLM
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-pro-exp-02-05")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")  # Read from env with fallback
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 256))
TOP_K = int(os.getenv("TOP_K", 5))

# Initialize Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize LLM
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)

# Initialize embeddings with Gemini model
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
    task_type="retrieval_document"
)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize or get Pinecone index
def init_pinecone():
    # Check if index exists, if not create it
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,  # Google embedding-001 dimension is 768
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    # Connect to the index
    return pc.Index(PINECONE_INDEX_NAME)

# Get the Langchain vectorstore
def get_vectorstore():
    index = init_pinecone()
    return LCPinecone(index, embeddings, "text")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_file)
        temp_file.flush()
        
        try:
            reader = PdfReader(temp_file.name)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        finally:
            os.unlink(temp_file.name)

# Function to split text into chunks
def split_text(text: str, filename: str) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 4,  # Approximate characters per word
        chunk_overlap=20,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(text)
    
    # Convert to Document objects with metadata
    return [
        Document(page_content=chunk, metadata={"source": filename})
        for chunk in chunks
    ]

# Function to store chunks in vector database
async def store_embeddings(chunks: List[Document]) -> None:
    try:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        return len(chunks)
    except Exception as e:
        logger.error(f"Error embedding content: {str(e)}")
        # Check if it's a model format error
        if "unexpected model name format" in str(e):
            logger.error(f"The embedding model name '{EMBEDDING_MODEL}' seems to be incorrect. "
                        f"Please check your EMBEDDING_MODEL configuration.")
        raise

# Function to perform semantic search
async def semantic_search(query: str) -> List[Dict[str, Any]]:
    try:
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search_with_score(
            query, 
            k=TOP_K
        )
        
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "score": score
            }
            for doc, score in results
        ]
    except Exception as e:
        logger.error(f"Error during semantic search: {str(e)}")
        # Check if it's a model format error
        if "unexpected model name format" in str(e):
            logger.error(f"The embedding model name '{EMBEDDING_MODEL}' seems to be incorrect. "
                        f"Please check your EMBEDDING_MODEL configuration.")
        raise

# Function to generate response from LLM
async def generate_response(query: str, context_docs: List[Dict[str, Any]]) -> str:
    context = "\n\n".join([f"Document: {doc['content']}" for doc in context_docs])
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided document contexts.
    
Context:
{context}

Question: {query}

Please provide a detailed and accurate answer based only on the information in the context above. If the context doesn't provide enough information to answer the question properly, acknowledge that and explain what's missing."""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Sorry, I encountered an error while generating a response: {str(e)}"

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "Hi! I'm your PDF RAG bot. Upload PDF files, and I'll answer questions about them.\n"
        "Use /help to see available commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Here's what I can do:\n"
        "1. Upload a PDF file, and I'll process it for future questions.\n"
        "2. Ask me any question about your uploaded PDFs, and I'll provide answers with sources.\n"
        "\nCommands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message"
    )

# Handle PDF files
async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process uploaded PDF files."""
    user_id = update.message.from_user.id
    
    # Get the file
    pdf_file = await context.bot.get_file(update.message.document.file_id)
    pdf_bytes = await pdf_file.download_as_bytearray()
    pdf_io = io.BytesIO(pdf_bytes)
    
    # Get the filename
    filename = update.message.document.file_name
    
    await update.message.reply_text(f"Processing PDF: {filename}...")
    
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text.strip():
            await update.message.reply_text("Could not extract any text from this PDF. It may be scanned or protected.")
            return
            
        # Split into chunks
        chunks = split_text(text, filename)
        
        # Store in vector database
        chunk_count = await store_embeddings(chunks)
        
        await update.message.reply_text(
            f"✅ PDF processed successfully!\n"
            f"• File: {filename}\n"
            f"• Chunks created: {chunk_count}\n"
            f"• Stored in vector database\n\n"
            f"You can now ask questions about this document."
        )
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        await update.message.reply_text(f"Error processing PDF: {str(e)}")

# Handle text messages (queries)
async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process user queries."""
    query = update.message.text
    user_id = update.message.from_user.id
    
    # Check if message is a command
    if query.startswith('/'):
        return
        
    await update.message.reply_text("Searching for relevant information...")
    
    try:
        # Perform semantic search
        results = await semantic_search(query)
        
        if not results:
            await update.message.reply_text(
                "I couldn't find any relevant information in your documents. "
                "Try rephrasing your question or upload more documents."
            )
            return
            
        # Generate response
        response = await generate_response(query, results)
        
        # Format sources
        sources = set(doc["source"] for doc in results)
        sources_text = "\n".join([f"• {source}" for source in sources])
        
        # Send response with sources
        await update.message.reply_text(
            f"{response}\n\n"
            f"Sources:\n{sources_text}"
        )
    except Exception as e:
        logger.error(f"Error handling query: {e}")
        await update.message.reply_text(f"Error processing your query: {str(e)}")

def main() -> None:
    """Start the bot."""
    # Initialize Pinecone
    init_pinecone()
    
    # Create the Application and pass it your bot's token
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Add message handlers
    application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main() 