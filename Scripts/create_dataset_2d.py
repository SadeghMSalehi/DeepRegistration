import os

import click
import numpy as np

from ..util import (
    load_3d_data,
    create_affine_matrix,
    create_rotation_matrix,
    similarity_transform_volumes,
    create_rotation_matrix,
    vec5_to_vec3,
    vrrotmat2vec,
    auto_crop,
    
)

data_path = './resources/Data/'
target_size = [120, 120, 120]
sample_number = 25

files = os.listdir(data_path+'controls/')
number_of_files = np.size(files)
for j in range(65):
    dataset = np.zeros([number_of_files*sample_number,
                        target_size[0],
                        target_size[1],
                        target_size[2]])
    labels = np.zeros([number_of_files*sample_number, 3])


    for counter, file_name in enumerate(files):

        Image, Image_affine = load_3d_data(data_path+'controls/'+file_name)
        croped_image = auto_crop(Image)

        for i in range(sample_number):
            affine, rotation = create_affine_matrix([1,1],
                                                    [-180,179],
                                                    [0,0],
                                                    croped_image.shape)
            img_t, transform = similarity_transform_volumes(croped_image,
                                                            affine,
                                                            target_size,)

            rotation_matrix = create_rotation_matrix(rotation)
            vector = vrrotmat2vec(rotation_matrix)
            vec3 = vec5_to_vec3(vector)

            dataset[sample_number*counter+i, ...] = img_t
            labels[sample_number*counter+i, ...] = vec3
    print(j)

    np.save('/host/data/180DegreeData/imageData'+str(j),dataset)
    np.save('/host/data/180DegreeData/Label'+str(j),labels)