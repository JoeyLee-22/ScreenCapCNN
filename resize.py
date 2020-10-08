import numpy as np

class resize:
    def resize(image, new_height, new_width):
        new = np.empty([new_height,new_width,3])
        factor = image.shape[0]/new_height

        for row in range (image.shape[0]):
            for column in range (image.shape[1]):
                if row%factor == 0 and column%factor == 0:
                    new[int(row/factor)][int(column/factor)] = image[row][column]

        return new