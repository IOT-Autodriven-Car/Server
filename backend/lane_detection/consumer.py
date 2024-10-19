import numpy as np
import cv2
from channels.generic.websocket import AsyncWebsocketConsumer
import json


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
                    await self.send(text_data=json.dumps({'result': result}))
                else:
                    await self.send(text_data=json.dumps({'result': None}))
            else:
                print("Failed to decode frame")

    def process_frame(self, frame):
        return "Processed frame"
