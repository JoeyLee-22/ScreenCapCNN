from screen_cap import start
from CNN import convolutional_neural_network

new_height,new_width = 300,480

cnn = convolutional_neural_network()
cnn.data_prep(new_height, new_width)
cnn.run(epochs=25)

start(cnn, new_height, new_width, minutes=0.1)