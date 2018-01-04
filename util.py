import nibabel as nib
import nilearn.image as nil_image
import numpy as np
import SimpleITK as sitk

def load_3d_data(
    filename,
):
    """Loads data.
    """
    img = nib.load(filename)

    volume = img.get_data()
    affine = img.affine

    return volume, affine


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
    rot = np.matmul(Rz, Ry) 
    rot = np.matmul(rot, Rx)

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

        return affine_trans, rotation


def similarity_transform_volumes(
    image,
    affine_trans,
    target_size,
    interpolation = 'continuous',
):
    image_size = np.shape(image)
    possible_scales = np.true_divide(image_size, target_size)
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


def vec3_to_vec5(vec3):
    teta = np.linalg.norm(vec3)
    vec = vec3/teta
    vec5 = np.zeros((5,1))
    vec5[0] = vec[0]
    vec5[1] = vec[1]
    vec5[2] = vec[2]
    vec5[3] = teta
    vec5[4] = 1
    
    return vec5


def vec5_to_vec3(vec5):
    return vec5[3,0]*vec5[:3,0]


def vrrotmat2vec(mat1, rot_type='proper'):
    """
    Create an axis-angle np.array from Rotation Matrix:
    ====================

    @param mat:  The nx3x3 rotation matrices to convert
    @type mat:   nx3x3 numpy array

    @param rot_type: 'improper' if there is a possibility of
                      having improper matrices in the input,
                      'proper' otherwise. 'proper' by default
    @type  rot_type: string ('proper' or 'improper')

    @return:    The 3D rotation axis and angle (ax_ang)
                5 entries:
                   First 3: axis
                   4: angle
                   5: 1 for proper and -1 for improper
    @rtype:     numpy 5xn array

    """
    mat = np.copy(mat1)
    if mat.ndim == 2:
        if np.shape(mat) == (3, 3):
            mat = np.copy(np.reshape(mat, (1, 3, 3)))
        else:
            raise Exception('Wrong Input Type')
    elif mat.ndim == 3:
        if np.shape(mat)[1:] != (3, 3):
            raise Exception('Wrong Input Type')
    else:
        raise Exception('Wrong Input Type')

    msz = np.shape(mat)[0]
    ax_ang = np.zeros((5, msz))

    epsilon = 1e-12
    if rot_type == 'proper':
        ax_ang[4, :] = np.ones(np.shape(ax_ang[4, :]))
    elif rot_type == 'improper':
        for i in range(msz):
            det1 = np.linalg.det(mat[i, :, :])
            if abs(det1 - 1) < epsilon:
                ax_ang[4, i] = 1
            elif abs(det1 + 1) < epsilon:
                ax_ang[4, i] = -1
                mat[i, :, :] = -mat[i, :, :]
            else:
                raise Exception('Matrix is not a rotation: |det| != 1')
    else:
        raise Exception('Wrong Input parameter for rot_type')



    mtrc = mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2]


    ind1 = np.where(abs(mtrc - 3) <= epsilon)[0]
    ind1_sz = np.size(ind1)
    if np.size(ind1) > 0:
        ax_ang[:4, ind1] = np.tile(np.array([0, 1, 0, 0]), (ind1_sz, 1)).transpose()


    ind2 = np.where(abs(mtrc + 1) <= epsilon)[0]
    ind2_sz = np.size(ind2)
    if ind2_sz > 0:
        # phi = pi
        # This singularity requires elaborate sign ambiguity resolution

        # Compute axis of rotation, make sure all elements >= 0
        # real signs are obtained by flipping algorithm below
        diag_elems = np.concatenate((mat[ind2, 0, 0].reshape(ind2_sz, 1),
                                     mat[ind2, 1, 1].reshape(ind2_sz, 1),
                                     mat[ind2, 2, 2].reshape(ind2_sz, 1)), axis=1)
        axis = np.sqrt(np.maximum((diag_elems + 1)/2, np.zeros((ind2_sz, 3))))
        # axis elements that are <= epsilon are set to zero
        axis = axis*((axis > epsilon).astype(int))

        # Flipping
        #
        # The algorithm uses the elements above diagonal to determine the signs
        # of rotation axis coordinate in the singular case Phi = pi.
        # All valid combinations of 0, positive and negative values lead to
        # 3 different cases:
        # If (Sum(signs)) >= 0 ... leave all coordinates positive
        # If (Sum(signs)) == -1 and all values are non-zero
        #   ... flip the coordinate that is missing in the term that has + sign,
        #       e.g. if 2AyAz is positive, flip x
        # If (Sum(signs)) == -1 and 2 values are zero
        #   ... flip the coord next to the one with non-zero value
        #   ... ambiguous, we have chosen shift right

        # construct vector [M23 M13 M12] ~ [2AyAz 2AxAz 2AxAy]
        # (in the order to facilitate flipping):    ^
        #                                  [no_x  no_y  no_z ]

        m_upper = np.concatenate((mat[ind2, 1, 2].reshape(ind2_sz, 1),
                                  mat[ind2, 0, 2].reshape(ind2_sz, 1),
                                  mat[ind2, 0, 1].reshape(ind2_sz, 1)), axis=1)

        # elements with || smaller than epsilon are considered to be zero
        signs = np.sign(m_upper)*((abs(m_upper) > epsilon).astype(int))

        sum_signs = np.sum(signs, axis=1)
        t1 = np.zeros(ind2_sz,)
        tind1 = np.where(sum_signs >= 0)[0]
        t1[tind1] = np.ones(np.shape(tind1))

        tind2 = np.where(np.all(np.vstack(((np.any(signs == 0, axis=1) == False), t1 == 0)), axis=0))[0]
        t1[tind2] = 2*np.ones(np.shape(tind2))

        tind3 = np.where(t1 == 0)[0]
        flip = np.zeros((ind2_sz, 3))
        flip[tind1, :] = np.ones((np.shape(tind1)[0], 3))
        flip[tind2, :] = np.copy(-signs[tind2, :])

        t2 = np.copy(signs[tind3, :])

        shifted = np.column_stack((t2[:, 2], t2[:, 0], t2[:, 1]))
        flip[tind3, :] = np.copy(shifted + (shifted == 0).astype(int))

        axis = axis*flip
        ax_ang[:4, ind2] = np.vstack((axis.transpose(), np.pi*(np.ones((1, ind2_sz)))))

    ind3 = np.where(np.all(np.vstack((abs(mtrc + 1) > epsilon, abs(mtrc - 3) > epsilon)), axis=0))[0]
    ind3_sz = np.size(ind3)
    if ind3_sz > 0:
        phi = np.arccos((mtrc[ind3]-1)/2)
        den = 2*np.sin(phi)
        a1 = (mat[ind3, 2, 1]-mat[ind3, 1, 2])/den
        a2 = (mat[ind3, 0, 2]-mat[ind3, 2, 0])/den
        a3 = (mat[ind3, 1, 0]-mat[ind3, 0, 1])/den
        axis = np.column_stack((a1, a2, a3))
        ax_ang[:4, ind3] = np.vstack((axis.transpose(), phi.transpose()))

    return ax_ang


def geodesic_distance(
    rot1,
    rot2,
):
    
    distance = np.abs(
        np.arccos(
            np.clip(
                0.5*np.trace(
                    np.matmul(
                        np.transpose(rot1),
                        rot2,
                    )
                )-1,
                -1.0+1e-7,
                1.0-1e-7,
            )
        )
    )

    return distance
