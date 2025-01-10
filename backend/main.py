from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from fastapi.websockets import WebSocket
import asyncio
from models import recognize_image
from dotenv import load_dotenv  
import os
from icecream import ic


load_dotenv()

app = FastAPI()



origins = [
    "http://localhost:5173",
    "http://localhost:5173/",
    "http://localhost:3000",
    "http://localhost:3000/",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173/",
    "http://127.0.0.1:3000/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            result = recognize_image(frame, model_path=os.getenv('MODEL_PATH'), embeddings_path=os.getenv('EMBEDDINGS_PATH'), class_mapping_path=os.getenv('CLASS_MAPPING_PATH'))
            ic(result)
            
            await websocket.send_json({"recognition": result})
            await asyncio.sleep(5)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
