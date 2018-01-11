import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from datetime import datetime, timedelta

import click
import numpy as np
from keras.optimizers import SGD
import keras

import resnet
from .util import (
    load_3d_data,
    create_affine_matrix,
    create_rotation_matrix,
    similarity_transform_volumes,
    create_rotation_matrix,
    vec3_to_vec5,
    vrrotvec2mat,
    geodesic_distance,
    rot_distance,
)
from losses import geodesic
# -----------------------------------------------------------------------------
width = 120
height = 120
#depth = 120
n_channels = 1
nb_classes = 3

batch_size = 32

# -----------------------------------------------------------------------------

def train_model():
    model = resnet.ResnetBuilder.build_resnet_18((n_channels, width, height), nb_classes)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    current_time = datetime.now() + timedelta(hours=-5)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='/host/home/exx/Documents/tensorboard/'+str(current_time)[:19], histogram_freq=0, write_graph=True, write_images=True)

    for j in range(30):
        for i in range(65):
            images = np.load('/host/data/30DegreeData/imageData'+str(i)+'.npy')
            labels = np.load('/host/data/30DegreeData/Label'+str(i)+'.npy')
            print('data loaded')

            img2feed = np.zeros((900, width, height, 30))
            for dim in range(1, 4):
                random_slices = np.random.randint(50, 80, 10)
                tmp = np.moveaxis(images, dim, 1)
                tmp = tmp[..., random_slices]
                img2feed[..., (dim-1)*10:dim*10] = tmp

            img2feed = np.moveaxis(img2feed, -1, 1)
            img2feed = np.reshape(img2feed, (-1 ,width, height, n_channels))
            labels = np.reshape(np.tile(labels,30), (-1, 3))
            del images
            print('data rearranged')
            
            model.fit(img2feed[:-(75*30)],
                      labels[:-(75*30)],
                      batch_size=32,
                      epochs=(65*10*j)+10*i+10,
                      initial_epoch = (65*10*j)+10*i,
                      callbacks=[tbCallBack],
                      validation_data=(img2feed[-(75*30):], labels[-(75*30):]),
                      shuffle=True)
            model.save('/host/home/exx/Documents/resources/model/3view_2dTo3d_4.h5')
    
    return

if __name__ == '__main__':
    train_model()