from django.urls import path
from core import consumers

websocket_urlpatterns = [
    path('ws/some_path/', consumers.YourConsumer.as_asgi()),
]
