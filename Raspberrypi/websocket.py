import asyncio
import websockets
import cv2
import numpy as np
import base64


async def send_image(uri):
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot access camera")
                break
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            await websocket.send(jpg_as_text)
            print("Send successfully")
            response = await websocket.recv()
            print(f"Receive: {response}")

            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

uri = "ws://localhost:8000/ws/stream/"
asyncio.get_event_loop().run_until_complete(send_image(uri))
