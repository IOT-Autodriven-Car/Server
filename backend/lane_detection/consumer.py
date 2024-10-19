import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
import json


class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # Giả sử bytes_data là frame ảnh (JPEG), bạn có thể dùng OpenCV để xử lý
            np_arr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Tùy theo xử lý bạn có thể trả về chuỗi hoặc không trả về
            result = self.process_frame(frame)  # Đổi thành self.process_frame

            if result:
                await self.send(text_data=json.dumps({'result': result}))
            else:
                await self.send(text_data=json.dumps({'result': None}))

    def process_frame(self, frame):
        # Thực hiện xử lý frame ở đây, ví dụ trả về chuỗi thông tin
        # Bạn có thể phân tích hình ảnh, phát hiện khuôn mặt, OCR, v.v...
        return "Processed frame"
