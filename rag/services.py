from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader
import docx
import redis
import re
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai  # Correct import
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain.prompts import HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate
import time
import chromadb
import shutil
from concurrent.futures import ThreadPoolExecutor
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
os.environ['GOOGLE_API_KEY'] = api_key

# Initialize embeddings and LLM model
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
LLM = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.4, max_output_tokens=512)

# Setup Redis connection
REDISCONNECTION = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
REDISCONNECTION.flushdb()

# Timer to track connection time
connection_start = time.time()
connection_end = time.time()
print(f"Connection time : {connection_end - connection_start}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Function to extract text from TXT file
def extract_text_from_txt(txt_file):
    text = txt_file.read().decode('utf-8')
    return text

# Function to handle document uploads and splitting into chunks
def handle_documents(paths, chunk_size, chunk_overlap):
    texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for path in paths:
        ext = path.name.split(".")[-1]
        if ext == 'docx':
            texts.extend(text_splitter.split_text(extract_text_from_docx(path)))
        if ext == 'pdf':
            texts.extend(text_splitter.split_text(extract_text_from_pdf(path)))
        if ext == 'txt':
            texts.extend(text_splitter.split_text(extract_text_from_txt(path)))
    return texts

# Function to load documents and store them in vector database
def load_documents(user, paths, chunk_size=2000, chunk_overlap=500):
    texts = handle_documents(paths, chunk_size, chunk_overlap)
    with ThreadPoolExecutor(max_workers=5) as executor:
        future = executor.submit(store_in_db, user, texts)
        future.result()

# Function to store text data in vector store (ChromaDB)
def store_in_db(user, texts):
    print(f"Loading to database for user: {user}")
    vector_store = get_vector_store(user)
    vector_store.add_texts(texts)
    print(f"Documents stored successfully in Database for user: {user}")

# Function to get vector store instance
def get_vector_store(user):
    chroma_vector_store = Chroma(
        persist_directory=user + "_db",
        collection_name='vector_index',
        embedding_function=EMBEDDING_MODEL,
    )
    return chroma_vector_store

# Function to create history-aware retriever for contextual conversations
def conversational_retriever(retriever):
    system_message_for_contextualization = SystemMessagePromptTemplate(
        prompt=PromptTemplate(template=(
          "You are a fashion assistant helping users design outfits based on available stock. "
                "When a user asks for outfit suggestions, ask follow-up questions to understand their preferences Display these questions in mcq format like A) and so on"
                "such as occasion, style, and types. Use the chat history to maintain context"
  )
    ))
    prompt_for_contextualization = ChatPromptTemplate(
        [
            system_message_for_contextualization,
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate(prompt=PromptTemplate(template="{input}", input_variables=['input']), input_variables=['input'])
        ],
        input_variables=['input']
    )
    history_aware_retriever = create_history_aware_retriever(LLM, retriever, prompt_for_contextualization)
    return history_aware_retriever

# Function to create conversational chain for cross-questioning and recommendations
def conversational_chain(history_aware_retriever):
    system_message_for_qa = SystemMessagePromptTemplate(prompt=PromptTemplate(
        template="You are a virtual sales assistant from Mostkino, a men's bottom wear apparel brand. Provide personalized assistance to help users find the best product for their needs. Be friendly, approachable, and make suggestions as if you are a real in-store sales assistant. Engage the user by asking follow-up questions when appropriate ask the user preferences in mcq format like A) etc while provide minimum 4 suggestions (in not more than 3 words ) dont ask question  dont make suggestions at first ask user their style vibe occasion color all these sort of preferences then display from the retrival"
 ))
    
    human_message_for_qa = HumanMessagePromptTemplate(
        prompt=PromptTemplate(template="Available Options: {context} \nUser Question : {input} \nRespond Friendly in mcq format like A) etc:",
                              input_variables=['input', 'context']), input_variables=['input', 'context']
    )
    
    
    prompt_for_qa = ChatPromptTemplate(
        [
            system_message_for_qa,
            MessagesPlaceholder('chat_history'),
            human_message_for_qa
        ],
        input_variables=['input', 'chat_history', 'context']
    )
    
    qa_chain = create_stuff_documents_chain(LLM, prompt_for_qa)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain

# Function to handle the chatbot interaction and store suggestions in variables
def chat(user, prompt, retriever):
    if not REDISCONNECTION.exists(f"chat_history:{user}"):
        REDISCONNECTION.set(f"chat_history:{user}", json.dumps([]))
    chat_history_json = REDISCONNECTION.get(f"chat_history:{user}")
    chat_history = json.loads(chat_history_json) if chat_history_json else []
    
    # Load clothing data from TXT file into the retriever
    history_aware_retriever = conversational_retriever(retriever)
    rag_chain = conversational_chain(history_aware_retriever)
    
    # Invoke the chain to get the bot's answer
    ai_message = rag_chain.invoke({'input': prompt, 'chat_history': chat_history})
    answer = ai_message['answer']
    
    # Insert newline characters after full stops, question marks, and exclamation marks
    formatted_answer = answer.replace(". ", ".\n").replace("? ", "?\n").replace("! ", "!\n")
    
    # Update the chat history in Redis
    chat_history.extend([
        ('human', prompt),
        ('ai', formatted_answer)
    ])
    REDISCONNECTION.set(f"chat_history:{user}", json.dumps(chat_history))
    
    # Extract suggestions and cross questions into variables
    suggestions = []
    cross_questions = []
    lines = formatted_answer.split("\n")
    pattern = r'([a-zA-Z])\)\s*(.+)'
    matches = re.findall(pattern, answer)

    # Creating a dictionary of options
    suggestions = [match[1].strip() for match in matches]
    print(suggestions)
    for line in lines:
        # if "suggest" in line.lower():
        #     suggestions.append(line)
        if "?" in line:
            cross_questions.append(line)
    
    # Format the final response as per your request
    response = {
        "question": prompt,
        "answer": formatted_answer,
        "suggestions": suggestions,
        "cross_questions": cross_questions
    }
    
    # Print formatted response for debugging (optional)
    print(json.dumps(response, indent=4))
    
    return response
