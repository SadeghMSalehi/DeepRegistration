import os

import click
import numpy as np
import nibabel as nib
import pandas as pd

from .model_3d import model_3d
from .util import (
    load_3d_data,
    create_affine_matrix,
    create_rotation_matrix,
    similarity_transform_volumes,
    create_rotation_matrix,
    vec3_to_vec5,
    vec5_to_vec3,
    vrrotvec2mat,
    vrrotmat2vec,
    geodesic_distance,
    auto_crop,
    rot_distance,
    
)

# -----------------------------------------------------------------------------
width = 120
height = 120
depth = 120
n_channels = 1 

number_of_eval = 100
model_path = './resources/model/3d_3d_4.h5'
data_path = './resources/Data/'
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
    target_size = [width, height, depth]
    
    df = pd.DataFrame()
    
    files = os.listdir(data_path+'Test/')
    number_of_files = np.size(files)
    for counter, file_name in enumerate(files):
        
        Image, Image_affine = load_3d_data(data_path+'Test/'+file_name)
        croped_image = auto_crop(Image)
        
        for i in range(10):
            for dim in range(1,4):
            
                affine_mov, rotation_mov = create_affine_matrix([1,1],
                                                        [-30,30],
                                                        [0,0],
                                                        croped_image.shape)

                TestImage[..., 0], _ = similarity_transform_volumes(croped_image,
                                                                    affine_mov,
                                                                    target_size,)

                GT_rot_mat = create_rotation_matrix(rotation_mov)
                vec5_GT = vrrotmat2vec(GT_rot_mat)
                vec3_GT = vec5_to_vec3(vec5_GT)
                inv_GT_rot_mat = np.linalg.inv(GT_rot_mat)

                vec3_pred = model.predict(TestImage, batch_size=1)
                vec5_pred = vec3_to_vec5(vec3_pred[0, :])
                pred_rot_mat = vrrotvec2mat(vec5_pred)
                inv_pred_rot_mat = np.linalg.inv(pred_rot_mat)
        
        
#                 affine_mov_pred, rotation_mov = create_affine_matrix([1,1],
#                                                                 inv_pred_rot_mat,
#                                                                 [0,0],
#                                                                 target_size)

#                 pred_rot_back_img, _ = similarity_transform_volumes(TestImage[0,:,:,:],
#                                                                     affine_mov_pred,
#                                                                     target_size,)
#                 nifti_image = nib.Nifti1Image(pred_rot_back_img, mov_affine)
#                 nib.save(nifti_image, './Pred_Backed'+str(i))
        
                Rot_distance2 = rot_distance(GT_rot_mat, pred_rot_mat)*180/3.1415
        
                df = df.append({'Subject Name': file_name,
                                'True rotation vector': vec3_GT,
                                'Predicted rotation vector': vec3_pred[0, :],
                                'MSE of rotation vectors': np.linalg.norm((vec3_pred[0, :]- vec3_GT)),
                                'Rot distance': Rot_distance2}, ignore_index=True)

                print("True rotation vector is: {}".format(vec3_GT))
                print("Predicted rotation vector is: {}".format(vec3_pred[0, :]))
                print("MSE of rotation vectors is: {}".format(np.linalg.norm((vec3_pred[0, :]- vec3_GT))))
                print("Rot distance is: {}".format(Rot_distance2))
                print('---------')
                
    df.to_csv('./3d_3d_4.csv')
    return

if __name__ == '__main__':
    eval_model(model_path, data_path)
        