from django.shortcuts import render
from rest_framework.response import Response
from .serializers import LaneImageSerializer
from rest_framework import status
from .models import LaneImageModel
from rest_framework.views import APIView

# Create your views with APIView.

class ImgUploadAPIView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        qs_serializer = LaneImageSerializer(
            data={
                "caption": request.data.get("caption"),
                "image": request.FILES.get("media"),
            },
            context={"request": request},
        )

        if qs_serializer.is_valid():
            qs_serializer.save()
            return Response(
                {
                    "message": "Media uploaded successfully.",
                    "data": qs_serializer.data,
                },
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                {"message": qs_serializer.errors, "data": None},
                status=status.HTTP_400_BAD_REQUEST,
            )

    def get(self, request):
        qs = LaneImageModel.objects.all()
        qs_serializer = LaneImageSerializer(qs, many=True)
        return Response(qs_serializer.data, status=status.HTTP_400_BAD_REQUEST)
