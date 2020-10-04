from screen_cap import run
from CNN import convolutional_neural_network

cnn = convolutional_neural_network()

cnn.data_prep()
cnn.run()
run(cnn, minutes=3)