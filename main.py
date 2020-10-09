from screen_cap import start
from CNN import convolutional_neural_network

new_height,new_width = 300,480

cnn = convolutional_neural_network(new_height, new_width)
cnn.data_prep()
cnn.run(epochs=25, train=True, plot=False)

start(cnn,new_height,new_width,minutes=0.1)