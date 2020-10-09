import numpy as np

def resize(image, new_height, new_width):
    new = np.empty([new_height,new_width,3])

    row_factor = int(image.shape[0]/new_height)
    column_factor = int(image.shape[1]/new_width)

    for row in range (0,image.shape[0], row_factor):
        for column in range (0,image.shape[1], column_factor):
            new[int(row/row_factor)][int(column/column_factor)] = image[row][column]

    return new