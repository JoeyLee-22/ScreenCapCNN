from screen_cap import start
from CNN import convolutional_neural_network

cnn = convolutional_neural_network()

cnn.data_prep()
cnn.run()
start(cnn, minutes=3)