from screen_cap import start
from CNN import convolutional_neural_network

cnn = convolutional_neural_network()
cnn.data_prep(resize_factor=1/7)
cnn.run(epochs=25)

start(cnn, minutes=0.1)