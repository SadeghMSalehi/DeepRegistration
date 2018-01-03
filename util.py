import numpy as np
import SimpleITK as sitk


def rotation_matrix(param):
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
    
    Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
    
    # Apply the rotation first around Y then X then Z.
    # To follow ITK transformation functions.
    rot = Rz @ Rx @ Ry 
    return rot