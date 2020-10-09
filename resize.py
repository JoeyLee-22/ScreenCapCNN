import numpy as np

def resize(image, new_height, new_width):
    new = np.empty([new_height,new_width,3])

    row_factor = int(image.shape[0]/new_height)
    column_factor = int(image.shape[1]/new_width)

    for row in range (new_height):
        for column in range (new_width):
            new[row][column] = image[row*row_factor][column*column_factor]

    return new