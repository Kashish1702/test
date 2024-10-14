import os
import time
import json
import redis
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import ChatGoogleGenerativeAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import AIMessage, HumanMessage
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key Loaded: {api_key}")
genai.configure(api_key=api_key)
os.environ['GOOGLE_API_KEY'] = api_key

# Initialize models and connections
connection_start = time.time()

EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
LLM = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
                             temperature=0.4,
                             max_output_tokens=512)
REDISCONNECTION = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
REDISCONNECTION.flushdb()
connection_end = time.time()
print(f"Connection time: {connection_end - connection_start} seconds")


def extract_text_from_pdf(pdf_file):
    from PyPDF2 import PdfReader
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(docx_file):
    import docx
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text_from_txt(txt_file):
    text = txt_file.read().decode('utf-8')
    return text


def handle_documents(paths, chunk_size, chunk_overlap):
    texts = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    for path in paths:
        ext = path.name.split(".")[-1]
        if ext == 'docx':
            texts.extend(text_splitter.split_text(extract_text_from_docx(path)))
        elif ext == 'pdf':
            texts.extend(text_splitter.split_text(extract_text_from_pdf(path)))
        elif ext == 'txt':
            with open(path, 'rb') as txt_file:
                texts.extend(text_splitter.split_text(extract_text_from_txt(txt_file)))
    return texts


def load_documents(user, paths, chunk_size=2000, chunk_overlap=500):
    texts = handle_documents(paths, chunk_size, chunk_overlap)
    with ThreadPoolExecutor(max_workers=5) as executor:
        future = executor.submit(store_in_db, user, texts)
        future.result()


def store_in_db(user, texts):
    print(f"Loading data into database for user: {user}")
    vector_store = get_vector_store(user)
    vector_store.add_texts(texts)
    print(f"Documents stored successfully in the database for user: {user}")


def get_vector_store(user):
    chroma_vector_store = Chroma(
        persist_directory=user + "_db",
        collection_name='vector_index',
        embedding_function=EMBEDDING_MODEL,
    )
    return chroma_vector_store


def extract_clothing_stock_data(txt_file):
    text = txt_file.read()
    return text


def load_clothing_stock(user, txt_file_path):
    with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
        data = extract_clothing_stock_data(txt_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(data)
    store_in_db(user, texts)


def conversational_retriever(retriever):
    system_message_for_contextualization = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=(
                "You are a fashion assistant helping users design outfits based on available stock. "
                "When a user asks for outfit suggestions, ask follow-up questions to understand their preferences "
                "such as occasion, style, colors, and sizes. Use the chat history to maintain context."
            ),
            input_variables=[]
        )
    )
    prompt_for_contextualization = ChatPromptTemplate(
        messages=[
            system_message_for_contextualization,
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template="{input}",
                    input_variables=['input']
                ),
                input_variables=['input']
            )
        ],
        input_variables=['input', 'chat_history']
    )
    history_aware_retriever = create_history_aware_retriever(LLM, retriever, prompt_for_contextualization)
    return history_aware_retriever


def conversational_chain(history_aware_retriever):
    system_message_for_qa = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=(
                "You are a fashion assistant helping users design outfits. Use the retrieved clothing stock information "
                "to suggest outfits. If you need more information, ask the user. Always suggest items that are available "
                "in stock. Be friendly and helpful."
            ),
            input_variables=[]
        )
    )
    human_message_for_qa = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=(
                "Context: {context}\n"
                "User: {input}\n"
                "Assistant:"
            ),
            input_variables=['input', 'context']
        ),
        input_variables=['input', 'context']
    )
    prompt_for_qa = ChatPromptTemplate(
        messages=[
            system_message_for_qa,
            MessagesPlaceholder('chat_history'),
            human_message_for_qa
        ],
        input_variables=['input', 'chat_history', 'context']
    )
    qa_chain = create_stuff_documents_chain(LLM, prompt_for_qa)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain


def chat(user, prompt, retriever):
    # Initialize chat history if not present
    if not REDISCONNECTION.exists(f"chat_history:{user}"):
        REDISCONNECTION.set(f"chat_history:{user}", json.dumps([]))
    chat_history_json = REDISCONNECTION.get(f"chat_history:{user}")
    chat_history = json.loads(chat_history_json) if chat_history_json else []
    # Create retriever and chain
    history_aware_retriever = conversational_retriever(retriever)
    rag_chain = conversational_chain(history_aware_retriever)
    print("Invoking chain...")
    # Invoke the chain with user input and chat history
    ai_message = rag_chain.invoke({'input': prompt, 'chat_history': chat_history})
    print("Chain invoked successfully")
    answer = ai_message['output']
    # Update chat history
    chat_history.extend([
        {'type': 'human', 'content': prompt},
        {'type': 'ai', 'content': answer}
    ])
    REDISCONNECTION.set(f"chat_history:{user}", json.dumps(chat_history))
    print("Chat history updated")
    return answer


if __name__ == "__main__":
    # Replace 'user1' with the actual user identifier
    user = 'user1'
    # Path to your clothing stock text file
    txt_file_path = 'clothing_stock.txt'
    # Load the clothing stock data into the vector store
    load_clothing_stock(user, txt_file_path)

    # Initialize the vector store retriever
    vector_store = get_vector_store(user)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Start the chat session
    print("Welcome to the Fashion Assistant Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Assistant: Thank you for using the Fashion Assistant. Have a great day!")
            break
        response = chat(user, user_input, retriever)
        print(f"Assistant: {response}")
