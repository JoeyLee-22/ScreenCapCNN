import numpy as np

def my_resize(image, new_height, new_width):
    new = np.empty([new_height,new_width,3])

    row_factor = int(image.shape[0]/new_height)
    column_factor = int(image.shape[1]/new_width)

    old_height = image.shape[0]
    old_width = image.shape[1]

    for row, old_row in zip(range(new_height), range(0,old_height,row_factor)):
        for column, old_column in zip(range(new_width), range(0,old_width,column_factor)):
            new[row][column]=image[old_row][old_column]

    return new