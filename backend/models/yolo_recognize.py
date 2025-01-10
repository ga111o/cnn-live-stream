from ultralytics import YOLO
import numpy as np
import asyncio
from icecream import ic
import time

_model = None
_embeddings = None
_class_mapping = None

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a, b) / (norm_a * norm_b)

def _load_models(model_path='yolov8n.pt', embeddings_path='yolo_embeddings.npy', class_mapping_path='yolo_class_mapping.npy'):
    global _model, _embeddings, _class_mapping
    
    if _model is None:
        try:
            _model = YOLO(model_path)
        except Exception as e:
            print(f"Warning: Error loading model: {e}")
            raise
            
        try:
            _embeddings = np.load(embeddings_path)
            _class_mapping = np.load(class_mapping_path, allow_pickle=True).item()
        except Exception as e:
            print(f"Warning: Error loading embeddings or class mapping: {e}")
            raise

async def recognize_image(image_path, model_path='yolov8n.pt', embeddings_path='yolo_embeddings.npy', class_mapping_path='yolo_class_mapping.npy'):
   
    if _model is None:
        _load_models(model_path, embeddings_path, class_mapping_path)
    
    if _model is None:
        raise RuntimeError("Failed to load YOLO model")
    
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, lambda: _model(image_path, verbose=False))
    
    if len(results[0].boxes.data) > 0:
        query_embedding = results[0].boxes.data[0].numpy()
        expected_dim = _embeddings.shape[1]
        if query_embedding.shape[0] != expected_dim:
            query_embedding = np.resize(query_embedding, expected_dim)
    else:
        query_embedding = np.zeros(_embeddings.shape[1])
    
    norm_query = np.linalg.norm(query_embedding)
    norm_embeddings = np.linalg.norm(_embeddings, axis=1)
    similarities = np.dot(_embeddings, query_embedding) / (norm_embeddings * norm_query + 1e-8)
    most_similar_idx = np.argmax(similarities)
    
    
    return {
        'class': _class_mapping[most_similar_idx],
        'similarity': float(similarities[most_similar_idx]),
    }


if __name__ == "__main__":
    _load_models()

    for i in range(1, 101):
        start_time = time.time()
        result = asyncio.run(recognize_image(f'./test_images/1.jpg'))
        end_time = time.time()
        time_ms = (end_time - start_time)*1000
        ic(i, result, time_ms)