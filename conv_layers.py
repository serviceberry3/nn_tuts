#Teaching myself about convolutional layers and filters in neural networks

	
#1D CONVOLUTION EXAMPLE
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv1D


#define input data - 1D array with 8 elements, with a bump in the middle
data = asarray([0, 0, 0, 1, 1, 0, 0, 0])


data = data.reshape(1, 8, 1)

#check that data was reshaped correctly
print(data)


#create model
model = Sequential()

model.add(Conv1D(1, 3, input_shape=(8, 1)))

# define a vertical line detector
weights = [asarray([[[0]],[[1]],[[0]]]), asarray([0.0])]

# store the weights in the model
model.set_weights(weights)

# confirm they were stored
print(model.get_weights())

# apply filter to input data
yhat = model.predict(data)
print(yhat)