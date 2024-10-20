import numpy as np
import cv2
from channels.generic.websocket import AsyncWebsocketConsumer
from .service.infer_frame import inferface_frame
import json
import os
from datetime import datetime
import threading



class VideoStreamConsumer(AsyncWebsocketConsumer):
    last_frame = None
    x_value = 0
    y_value = 0
    A = None
    B = None
    C = None
    D = None
    areaAB = None
    areaAC = None
    start_processing_frame = False

    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # np_arr = np.frombuffer(bytes_data, np.uint8)
            # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # if frame is not None:
            #     print("Frame received successfully")
            #     self.latest_frame = frame
            #     # THE_FRAME = frame.copy()
            #     # result = self.process_frame(THE_FRAME)
            #     # result_temp = "Hello"
            #     # if result_temp:
            #         # print(f"Sending result: {result}")
            #     await self.send(text_data=json.dumps({'result': f"{self.x_value},{self.y_value}"}))
            #     # else:
            #     #     await self.send(text_data=json.dumps({'result': None}))
            # thread1 = threading.Thread(target=self.get_frame, args=(bytes_data,))

            np_arr = np.frombuffer(bytes_data, np.uint8)
            self.last_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            print("Frame received successfully")
            # await self.send(text_data=json.dumps({'result': f"Image features:{self.A},{self.B},{self.C},{self.D},{self.areaAB},{self.areaAC}; Car Drive value:{self.x_value},{self.y_value}"}))
            await self.send(text_data=json.dumps({'Result': f"Image features:{self.A},{self.B},{self.C},{self.D},{self.areaAB},{self.areaAC}"}))

            
            thread1 = threading.Thread(target=self.process_frame)
            
            if self.start_processing_frame == False:
                thread1.start()
            # thread2.start()
            # thread1.join()

        else:
                print("Failed to decode frame")

    # async def get_frame(self, bytes_data):
            
    def process_frame(self):
        self.start_processing_frame = True
        # self.save_frame_to_image(self.last_frame)
        print("Hi")
        try:
            self.A, self.B, self.C, self.D, self.areaAB, self.areaAC =  inferface_frame(self.last_frame)
        except Exception as e:
            print(e)
        self.start_processing_frame = False
        return 0

    # def save_frame_to_image(self, frame):
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = f"frame_{timestamp}.jpg"

    #     save_dir = r"D:\IT_2\KTPM\IOT_BACKEND\Server\backend\lane_detection\service\outputs\images"
    #     os.makedirs(save_dir, exist_ok=True)

    #     save_path = os.path.join(save_dir, filename)

    #     cv2.imwrite(save_path, frame)

    #     print(f"Frame saved to {save_path}")
    #     return save_path