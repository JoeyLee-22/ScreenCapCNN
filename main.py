from screen_cap import start
from CNN import convolutional_neural_network

(new_height,new_width) = (350, 560) 

cnn = convolutional_neural_network(new_height, new_width)
cnn.run(epochs=50, train=True, evaluate=True, plot=False, data_prep=True, clear_data=True)

start(cnn, new_height, new_width, minutes=0.1)