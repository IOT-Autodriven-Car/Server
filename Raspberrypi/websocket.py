import asyncio
import cv2
import websockets


async def send_frames(websocket):
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot access camera")
            break

        _, buffer = cv2.imencode('.jpg', frame)

        await websocket.send(buffer.tobytes())
        print("Frame sent successfully")

        # Hiện frame (nếu cần)
        # cv2.imshow('Camera', frame)

        await asyncio.sleep(0.1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


async def receive_response(websocket):
    while True:
        try:
            response = await websocket.recv()
            print(f"Received: {response}")
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
            break


async def send_image(uri):
    async with websockets.connect(uri) as websocket:
        send_task = asyncio.create_task(send_frames(websocket))
        receive_task = asyncio.create_task(receive_response(websocket))

        await asyncio.gather(send_task, receive_task)

uri = "ws://localhost:8000/ws/stream/"
asyncio.get_event_loop().run_until_complete(send_image(uri))
