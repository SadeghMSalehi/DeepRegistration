import os

import click
import numpy as np
import nibabel as nib

import resnet
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
    
)

# -----------------------------------------------------------------------------
width = 120
height = 120
depth = 120
n_channels = 1 

number_of_eval = 100
model_path = '/host/home/exx/Documents/resources/model/2dTo3d_2.h5'
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
    model = resnet.ResnetBuilder.build_resnet_18((n_channels, width, height), 3)    
    model.load_weights(model_path)

    TestImage = np.zeros((1, width, height, depth))
    result = np.zeros((number_of_eval, 4))
    target_size = [width, height, depth]

    ref_img, ref_affine = load_3d_data(os.path.join(data_path, 'nT29template.nii'))
    mov_img, mov_affine = load_3d_data(os.path.join(data_path, '29_c160_f234.nii.gz'))

    croped_image = auto_crop(mov_img)
    
    for i in range(number_of_eval):

        affine_mov, rotation_mov = create_affine_matrix([1,1],
                                                        [-30,30],
                                                        [0,0],
                                                        croped_image.shape)

        TestImage[0:1,...], _ = similarity_transform_volumes(croped_image,
                                                            affine_mov,
                                                            target_size,)

        nifti_image = nib.Nifti1Image(TestImage[0,...], mov_affine)
        nib.save(nifti_image, './rotated'+str(i))
        
        GT_rot_mat = create_rotation_matrix(rotation_mov)
        vec5_GT = vrrotmat2vec(GT_rot_mat)
        vec3_GT = vec5_to_vec3(vec5_GT)
        inv_GT_rot_mat = np.linalg.inv(GT_rot_mat)

        vec3_pred = model.predict(TestImage[...,60:61], batch_size=1)
        vec5_pred = vec3_to_vec5(vec3_pred[0, :])
        pred_rot_mat = vrrotvec2mat(vec5_pred)
        inv_pred_rot_mat = np.linalg.inv(pred_rot_mat)
        
        affine_mov, rotation_mov = create_affine_matrix([1,1],
                                                        inv_GT_rot_mat,
                                                        [0,0],
                                                        target_size)

        GT_rot_back_img, _ = similarity_transform_volumes(TestImage[0,:,:,:],
                                                            affine_mov,
                                                            target_size,)
        nifti_image = nib.Nifti1Image(GT_rot_back_img, mov_affine)
        nib.save(nifti_image, './GT_Backed'+str(i))
        
        affine_mov, rotation_mov = create_affine_matrix([1,1],
                                                        inv_pred_rot_mat,
                                                        [0,0],
                                                        target_size)

        pred_rot_back_img, _ = similarity_transform_volumes(TestImage[0,:,:,:],
                                                            affine_mov,
                                                            target_size,)
        nifti_image = nib.Nifti1Image(pred_rot_back_img, mov_affine)
        nib.save(nifti_image, './Pred_Backed'+str(i))
        
        distance = geodesic_distance(GT_rot_mat, pred_rot_mat)
#         result[i,:-1] = rotation_mov
        result[i,-1] = distance
        print("True rotation vector is: {}".format(vec3_GT))
        print("Predicted rotation vector is: {}".format(vec3_pred[0, :]))
        print("MSE of rotation vectors is: {}".format(np.linalg.norm((vec3_pred[0, :]- vec3_GT))))
        print("Geodesic distance is: {}".format(distance))
        print('---------')
    return

if __name__ == '__main__':
    eval_model(model_path, data_path)
        