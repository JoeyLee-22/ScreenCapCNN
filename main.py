from screen_cap import start
from CNN import convolutional_neural_network

#screen size = (2100, 3360)
cnn = convolutional_neural_network(300, 480)
cnn.run(epochs=50, load_model=False, save_model=True, train=True, evaluate=True, plot=True, data_prep=True, clear_data=True)

start(cnn, minutes=0.1)