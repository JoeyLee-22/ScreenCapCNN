from screen_cap import start
from data_prep import data_prep
from CNN import convolutional_neural_network

(new_height,new_width) = (300,480)

data_prep(new_height, new_width)
cnn = convolutional_neural_network(new_height, new_width)
cnn.run(epochs=25, train=True, plot=False)

start(cnn, new_height, new_width, minutes=0.1)