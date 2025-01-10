from ultralytics import YOLO
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from icecream import ic

def train_model(image_dir='./images/'):
    if not os.path.exists(image_dir):
        raise FileNotFoundError("이미지 디렉토리가 존재하지 않습니다")

    # YOLO 모델 로드
    model = YOLO('yolov8n.pt')

    # 특징 추출을 위해 백본만 사용
    def extract_features(model, image_dir):
        embeddings = []
        class_names = []
        
        # 디렉토리 내의 모든 클래스 폴더 순회
        for class_name in sorted(os.listdir(image_dir)):
            class_path = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            # 각 클래스 폴더 내의 이미지 처리
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                # 특징 추출을 위해 백본 사용
                results = model(img_path, verbose=False)
                # boxes 데이터에서 특징 추출
                if len(results[0].boxes.data) > 0:
                    # 모든 감지된 객체의 특징을 평균
                    features = results[0].boxes.data.cpu().numpy().flatten()
                else:
                    # 감지된 객체가 없는 경우 0으로 채운 벡터 사용
                    features = np.zeros(6)  # boxes.data의 기본 차원
                embeddings.append(features)
                class_names.append(class_name)
                
        return np.array(embeddings), class_names

    embeddings, class_names = extract_features(model, image_dir)
    
    if len(embeddings) == 0:
        raise ValueError("처리된 이미지가 없습니다")

    unique_ids = np.arange(len(embeddings))
    class_mapping = {id: class_name for id, class_name in enumerate(sorted(set(class_names)))}
    
    # 결과 저장
    np.save('yolo_embeddings.npy', embeddings)
    np.save('yolo_unique_ids.npy', unique_ids)
    np.save('yolo_class_mapping.npy', class_mapping)
    
    return model, embeddings, unique_ids, class_mapping

if __name__ == "__main__":
    model, embeddings, unique_ids, class_mapping = train_model()
    ic("완료!")
