import os

import click
import numpy as np
from keras.optimizers import SGD

from .model_3d import model_3d
from .util import (
    load_3d_data,
    create_affine_matrix,
    create_rotation_matrix,
    similarity_transform_volumes,
    create_rotation_matrix,
    vec3_to_vec5,
    vrrotvec2mat,
    geodesic_distance,
    
)
from losses import geodesic
# -----------------------------------------------------------------------------
width = 120
height = 120
depth = 120
n_channels = 1

batch_size = 32

# -----------------------------------------------------------------------------

def train_model():
    model = model_3d(width, height, depth, n_channels)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=geodesic, optimizer=sgd)
    for j in range(30):
        for i in range(21):
            images = np.load('./imageData'+str(i)+'.npy')
            labels = np.load('./Label'+str(i)+'.npy')
            print('data loaded')
            images = images[..., np.newaxis]
            model.fit(images, labels, batch_size=32, epochs=100)
            model.save('./resources/model/Move_only.h5')
    
    return

if __name__ == '__main__':
    train_model()