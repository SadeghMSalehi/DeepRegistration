import os

import click
import numpy as np

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

# -----------------------------------------------------------------------------
width = 62
height = 76
depth = 54
n_channels = 2 

number_of_eval = 100
model_path = './resources/model/Reg_regression_Vector_pi_MSE_Keras4.h5'
data_path = './DeepRegistration/SampleData/' 
# -----------------------------------------------------------------------------

#@click.command()
#@click.argument('model_path', type=click.Path(exists=True, file_okay=False),
#                default='../resources/model/Reg_regression_Vector_pi_MSE_Keras4.h5')
#@click.argument('data_path', type=click.Path(exists=True, file_okay=False),
#                default='./SampleData/')


def eval_model(
    model_path,
    data_path,
):
    model = model_3d(width, height, depth, n_channels)
    model.load_weights(model_path)

    TestImage = np.zeros((1, width, height, depth, n_channels))
    result = np.zeros((number_of_eval, 4))
    target_size = [width, height, depth]

    ref_img, ref_affine = load_3d_data(os.path.join(data_path, 'nT29template.nii'))
    mov_img, mov_affine = load_3d_data(os.path.join(data_path, '29_c160_f234.nii.gz'))

    for i in range(number_of_eval):

        affine_mov, rotation_mov = create_affine_matrix([1,1],
                                                        [-100,100],
                                                        [0,0],
                                                        mov_img.shape)

        TestImage[..., 0], _ = similarity_transform_volumes(mov_img,
                                                            affine_mov,
                                                            target_size,)
        TestImage[..., 1], _ = similarity_transform_volumes(ref_img,
                                                            np.eye(4),
                                                            target_size,)

        GT_rot_mat = create_rotation_matrix(rotation_mov)

        vec3_pred = model.predict(TestImage, batch_size=1)
        vec5_pred = vec3_to_vec5(vec3_pred[0, :])

        pred_rot_mat = vrrotvec2mat(vec5_pred)

        distance = geodesic_distance(GT_rot_mat, pred_rot_mat)
        result[i,:-1] = rotation_mov
        result[i,-1] = distance
        print(distance)
        print(i)
    return

if __name__ == '__main__':
    eval_model(model_path, data_path)
        