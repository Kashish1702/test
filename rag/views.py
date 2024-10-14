from django.shortcuts import render
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import AIMessage,HumanMessage
from . import services
import time
from django.http import HttpResponse, JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
import os
import google.generativeai as genai

gemini_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=gemini_key)
os.environ['GOOGLE_API_KEY'] = gemini_key
@csrf_exempt
def upload_to_vector_store(request):
    start = time.time()
    user = request.POST['text']
    files = request.FILES.getlist('file')
    with ThreadPoolExecutor(max_workers = 5) as executor:
        future = executor.submit(services.load_documents, user, files)
        future.result()
    end_time = time.time()
    print(f"Time {end_time - start}")
    return JsonResponse({'message': f"Chroma db created for user {user} in time {end_time - start}"})

@csrf_exempt
def chat_with_docs(request):
    if request.method == 'POST':
        # Parse the JSON data from the request body
        data = json.loads(request.body)
        question = data.get('question')
        user = data.get('user')
        vector_store = services.get_vector_store(user)
        retriever = vector_store.as_retriever()
        answer = services.chat(user, question, retriever)
        print(f'reponse : {answer}')
        return JsonResponse({'question': question, 'answer': answer})

