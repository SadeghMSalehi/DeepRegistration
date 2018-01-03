import nilearn.image as nil_image
import numpy as np
import SimpleITK as sitk


def create_rotation_matrix(param):
    '''
    Create a rotation matrix from 3 rotation angels around X, Y, and Z:
    =================
    Arguments:
        param: numpy 1*3 array for [x, y, z] angels in degree.

    Output:
        rot: Correspond 3*3 rotation matrix rotated around y->x->z axises. 
    '''
    theta_x = param[0] * np.pi / 180
    cx = np.cos(theta_x)
    sx = np.sin(theta_x)

    theta_y = param[1] * np.pi / 180
    cy = np.cos(theta_y)
    sy = np.sin(theta_y)

    theta_z = param[2] * np.pi / 180
    cz = np.cos(theta_z)
    sz = np.sin(theta_z)
    
    Rx = [[1, 0, 0],
          [0, cx, -sx],
          [0, sx, cx]]

    Ry = [[cy, 0, sy],
          [0, 1, 0],
          [-sy, 0, cy]]

    Rz = [[cz, -sz, 0],
          [sz, cz, 0],
          [0, 0, 1]]
    
    # Apply the rotation first around Y then X then Z.
    # To follow ITK transformation functions.
    rot = np.matmul(Rz, Rx) 
    rot = np.matmul(rot, Ry)

    return rot


def create_affine_matrix(
    scale,
    rotation,
    translation,
    image_size,
):
        scale = np.random.uniform(scale[0], scale[1])
        rotation = np.random.uniform(rotation[0], rotation[1], 3)
        translation = np.random.uniform(translation[0], translation[1], 3)

        # Create rotation Matrix
        rot = create_rotation_matrix(rotation)

        affine_trans_rot = np.eye(4)
        affine_trans_rot[:3, :3] = rot

        # Create scale matrix
        affine_trans_scale = np.diag([scale, scale, scale, 1.])

        # Create translation matrix
        affine_trans_translation = np.eye(4)
        affine_trans_translation[:, 3] = [translation[0],
                                          translation[1],
                                          translation[2],
                                          1]

        # Create shift & unshift matrix to apply rotation around
        # center of image not (0,0,0)
        shift = - np.asarray(image_size) // 2
        affine_trans_shift = np.eye(4)
        affine_trans_shift[:, 3] = [shift[0],
                                    shift[1],
                                    shift[2],
                                    1]

        unshift = - shift
        affine_trans_unshift = np.eye(4)
        affine_trans_unshift[:, 3] = [unshift[0],
                                      unshift[1],
                                      unshift[2],
                                      1]

        # Apply transformations
        affine_trans = np.matmul(affine_trans_scale, affine_trans_translation)
        affine_trans = np.matmul(affine_trans, affine_trans_unshift)
        affine_trans = np.matmul(affine_trans, affine_trans_rot)
        affine_trans = np.matmul(affine_trans, affine_trans_shift)

        return affine_trans


def similarity_transform_volumes(
    image,
    affine_trans,
    target_size,
):
    image_size = np.shape(image)
    possible_scales = np.divide(image_size, target_size)
    crop_scale = np.max(possible_scales)
    if crop_scale <= 1:
        crop_scale = 1
    scale_transform = np.diag((crop_scale,
                               crop_scale,
                               crop_scale,
                               1))
    shift = -(
        np.asarray(target_size) - np.asarray(
            image_size // np.asarray(crop_scale),
        )
    ) // 2
    affine_trans_to_center = np.eye(4)
    affine_trans_to_center[:, 3] = [shift[0],
                                    shift[1],
                                    shift[2],
                                    1]

    transform = np.matmul(affine_trans, scale_transform)
    transform = np.matmul(transform, affine_trans_to_center)

    nifti_img = nib.Nifti1Image(image, affine=np.eye(4))
    nifti_image_t = nil_image.resample_img(
        nifti_img,
        target_affine=transform,
        target_shape=target_size,
        interpolation=interpolation,
    )
    image_t = nifti_image_t.get_data()
            
    return image_t, transform


def vrrotvec2mat(ax_ang):
    """
    Create a rotation matrix corresponding to the rotation around a general
    axis by a specified angle.
    """
    
    if ax_ang.ndim == 1:
        if np.size(ax_ang) == 5:
            ax_ang = np.reshape(ax_ang, (5, 1))
            msz = 1
        elif np.size(ax_ang) == 4:
            ax_ang = np.reshape(np.hstack((ax_ang, np.array([1]))), (5, 1))
            msz = 1
        else:
            raise Exception('Wrong Input Type')
    elif ax_ang.ndim == 2:
        if np.shape(ax_ang)[0] == 5:
            msz = np.shape(ax_ang)[1]
        elif np.shape(ax_ang)[1] == 5:
            ax_ang = ax_ang.transpose()
            msz = np.shape(ax_ang)[1]
        else:
            raise Exception('Wrong Input Type')
    else:
        raise Exception('Wrong Input Type')

    direction = ax_ang[0:3, :]
    angle = ax_ang[3, :]

    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d, axis=0)
    x = d[0, :]
    y = d[1, :]
    z = d[2, :]
    c = np.cos(angle)
    s = np.sin(angle)
    tc = 1 - c

    mt11 = tc*x*x + c
    mt12 = tc*x*y - s*z
    mt13 = tc*x*z + s*y

    mt21 = tc*x*y + s*z
    mt22 = tc*y*y + c
    mt23 = tc*y*z - s*x

    mt31 = tc*x*z - s*y
    mt32 = tc*y*z + s*x
    mt33 = tc*z*z + c

    mtx = np.column_stack((mt11, mt12, mt13, mt21, mt22, mt23, mt31, mt32, mt33))

    inds1 = np.where(ax_ang[4, :] == -1)
    mtx[inds1, :] = -mtx[inds1, :]

    if msz == 1:
        mtx = mtx.reshape(3, 3)
    else:
        mtx = mtx.reshape(msz, 3, 3)

    return mtx