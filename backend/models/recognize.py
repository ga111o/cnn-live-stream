import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors
import asyncio

_model = None
_nn_model = None
_class_mapping = None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _load_models(model_path='embedding_model.h5', embeddings_path='embeddings.npy', class_mapping_path='class_mapping.npy'):
    global _model, _class_mapping
    
    if _model is None:
        _model = tf.keras.models.load_model(model_path)
        global embeddings
        embeddings = np.load(embeddings_path)
        _class_mapping = np.load(class_mapping_path, allow_pickle=True).item()

async def recognize_image(image_path, model_path='embedding_model.h5', embeddings_path='embeddings.npy', class_mapping_path='class_mapping.npy'):
    _load_models(model_path, embeddings_path, class_mapping_path)
    
    # 이미지 전처리
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # tf.function을 루프 밖으로 이동
    @tf.function(experimental_relax_shapes=True)
    def predict(x):
        return _model(x, training=False)
    
    # 비동기 실행을 위해 run_in_executor 사용
    loop = asyncio.get_event_loop()
    query_embedding = await loop.run_in_executor(None, lambda: predict(img_array)[0].numpy())
    
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    most_similar_idx = np.argmax(similarities)
    
    return {
        'class': _class_mapping[most_similar_idx],
        'similarity': similarities[most_similar_idx]
    }

if __name__ == "__main__":
    import time
    from icecream import ic

    for i in range(1, 101):
        start_time = time.time()
        result = asyncio.run(recognize_image(f'./test_images/3.jpg'))
        end_time = time.time()
        time_ms = (end_time - start_time)*1000
        
        
        ic(i, result, time_ms)
