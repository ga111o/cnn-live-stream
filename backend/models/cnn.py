import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from icecream import ic

def train_model(image_dir='./images/'):
    if not os.path.exists(image_dir):
        raise FileNotFoundError("not os.path.exists(image_dir):")

    data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    data = data_gen.flow_from_directory(
        image_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode=None,
        shuffle=False
    )

    class_names = sorted(data.class_indices.keys())
    
    if data.samples == 0:
        raise ValueError("not data.samples == 0:")

    input_tensor = Input(shape=(150, 150, 3))
    x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(128, activation='relu')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    embeddings = model.predict(data)

    unique_ids = np.arange(len(embeddings))
    class_mapping = {id: class_name for id, class_name in enumerate(class_names)}
    
    nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn_model.fit(embeddings)

    np.save('embeddings.npy', embeddings)
    np.save('unique_ids.npy', unique_ids)
    np.save('class_mapping.npy', class_mapping)
    model.save('embedding_model.h5')

    return model, embeddings, unique_ids, class_mapping

if __name__ == "__main__":
    model, embeddings, unique_ids, class_mapping = train_model()
    ic("done!")
