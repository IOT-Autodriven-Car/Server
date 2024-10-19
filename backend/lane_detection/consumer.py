import numpy as np
import cv2
from channels.generic.websocket import AsyncWebsocketConsumer
from .service.infer_frame import inferface_frame
import json
import os
from datetime import datetime


class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            np_arr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                print("Frame received successfully")
                result = self.process_frame(frame)

                if result:
                    print(f"Sending result: {result}")
                    await self.send(text_data=json.dumps({'result': result}))
                else:
                    await self.send(text_data=json.dumps({'result': None}))
            else:
                print("Failed to decode frame")

    def process_frame(self, frame):
        return print(inferface_frame(frame))

    # def save_frame_to_image(self, frame):
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = f"frame_{timestamp}.jpg"

    #     save_dir = r"/home/vinhnado/Dev/IOT/Server/backend/lane_detection/img"
    #     os.makedirs(save_dir, exist_ok=True)

    #     save_path = os.path.join(save_dir, filename)

    #     cv2.imwrite(save_path, frame)

    #     print(f"Frame saved to {save_path}")
    #     return save_path
