import tensorflow as tf
from keras import backend as K
import numpy as np

def my_R(x):
    
    R1 = tf.eye(3) + tf.sin(x[2]) * x[0] + (1.0 - tf.cos(x[2])) * K.dot(x[0], x[0])
    R2 = tf.eye(3) + tf.sin(x[3]) * x[1] + (1.0 - tf.cos(x[3])) * K.dot(x[1], x[1])
    
    return K.dot(K.transpose(R1), R2)

# Rodrigues' formula
def get_theta(x):
    
    return tf.abs(
        tf.acos(
            tf.clip_by_value(
                0.5*(tf.reduce_sum(tf.diag_part(x))-1.0),
                -1.0+1e-7,
                1.0-1e-7)
        )
    )

def geodesic(y_pred, y_true):
    """ Geodesic Loss.

    Arguments:
        y_pred: `Tensor` of `float` type. Predicted values.
        y_true: `Tensor` of `float` type. Targets in degree. Should be transformed to 
        rotational matrix in order to calculate geodesic loss.

    """
    with tf.name_scope("Geodesic"):
        # compute geodesic viewpoint loss
        
        # compute angles
        angle_true = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=1))
        angle_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=1))
        
        # compute axes
        axis_true = tf.nn.l2_normalize(y_true, dim=1)
        axis_pred = tf.nn.l2_normalize(y_pred, dim=1)
        
        # convert axes to corresponding skew-symmetric matrices
        proj = tf.constant(np.asarray([
            [0, 0, 0, 0, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, -1, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0]]), dtype=tf.float32)
        
        skew_true = K.dot(axis_true, proj)
        skew_pred = K.dot(axis_pred, proj)
        skew_true = tf.map_fn(lambda x: tf.reshape(x, [3, 3]), skew_true)
        skew_pred = tf.map_fn(lambda x: tf.reshape(x, [3, 3]), skew_pred)
        
        # compute rotation matrices and do a dot product
        R = tf.map_fn(
            my_R,
            (skew_true,
             skew_pred,
             angle_true,
             angle_pred),
            dtype=tf.float32)
        
        # compute the angle error
        theta = tf.map_fn(get_theta, R)
        
        return tf.reduce_mean(theta)