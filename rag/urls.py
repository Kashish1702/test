from . import views
from django.urls import path,include

urlpatterns = [
    path("upload/", views.upload_to_vector_store,name = 'upload'),
    path("chat/",views.chat_with_docs,name = 'chat'),
]