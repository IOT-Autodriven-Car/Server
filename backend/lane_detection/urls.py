from django.urls import path
from .views import ImgUploadAPIView

app_name = "lane_detection"
urlpatterns = [path("media_upload/", ImgUploadAPIView.as_view(), name="media_upload")]
