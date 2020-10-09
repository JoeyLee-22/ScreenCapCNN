from screen_cap import start
from CNN import convolutional_neural_network

(new_height,new_width) = (300,480)

cnn = convolutional_neural_network(new_height, new_width)
cnn.run(epochs=50, train=True, plot=False, data_prep=False)

start(cnn, new_height, new_width, minutes=0.1)