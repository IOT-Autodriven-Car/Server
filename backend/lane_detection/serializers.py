from rest_framework import serializers
from .models import LaneImageModel

class LaneImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = LaneImageModel
        fields= ('id','caption','image')