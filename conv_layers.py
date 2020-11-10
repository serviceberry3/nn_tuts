#Teaching myself about convolutional layers and filters in neural networks

	
#1D CONVOLUTION EXAMPLE
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Conv2D

print("1D EXAMPLE")

#define input data - 1D array with 8 elements, with a bump in the middle
data = asarray([0, 0, 0, 1, 1, 0, 0, 0])

#input to Keras must be 3D.
#First dim refers to each input sample (only have one sample in this case)
#Second dim refers to len of each sample (8 in this case)
#Third dim refers to num of channels in each sample (just one in this case)
data = data.reshape(1, 8, 1)

#check that data was reshaped correctly
print(data)

#Define model that expects input samples to have shape [8, 1]
#Will have single filter w/shape of 3 (3 elements wide) 
#Keras refers to shape of filter as kernel_size

#create blank model
model = Sequential()

#add parameters to model: 1x3 filter, 8x1 input
model.add(Conv1D(1, 3, input_shape = (8, 1)))

#Filters in a conv layer usually initialized w/random weights. Here we'll manually specify weights for single filter. 
#We'll define filter capable of detecting bumps (high input vals surround by lows) - [0, 1, 0]

#Define a vertical line detector
#The conv layer also has bias input val that requires a weight that we will set to zero
#The wts must be specified in 3D structure (rows, columns, channels) The filter has single row, three columns, and one channel
weights = [asarray([[[0]],[[1]],[[0]]]), asarray([0.0])]

#Store the weights in the model
model.set_weights(weights)

#Confirm weights
print(model.get_weights())

#Apply filter to input data, returning the feature map directly (the output of applying filter systematically across input sequence)
yhat = model.predict(data)
print(yhat)

#Feature map is result of doing dot product as filter is dragged across input array
#W/different inputs, we may detect feature w/more or less intensity, and w/different weights in filter, we'd detect diff features in input sequence



#2D CONVOLUTION EXAMPLE - VERTICAL LINE DETECTOR IN 2D IMAGE
print("2D EXAMPLE")

#Define input data - a square 8×8 pixel input image with a single channel (e.g. grayscale) with a single vertical line in the middle.

#Input to a Conv2D layer must be 4D

#D1: the samples (only one in this case)
#D2: number of rows; in this case, eight
#D3: number of columns, again eight in this case
#D4: number of channels, which is one in this case

#So input must have 4D shape [samples, rows, columns, channels] [1, 8, 8, 1] in this case
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]

#add another dimension
data = asarray(data)
print(data)

#finalize shape
data = data.reshape(1, 8, 8, 1)
print(data)

#Create blank model
model = Sequential()

#Filter will be 2D and square with shape 3×3. The layer will expect input samples to have shape [columns, rows, channels] or [8,8,1]
model.add(Conv2D(1, (3,3), input_shape = (8, 8, 1)))

#Define filter: a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
            
#Finalize weight initialization with bias as 0
weights = [asarray(detector), asarray([0.0])]

#Store weights in the model
model.set_weights(weights)

#Confirm weights
print(model.get_weights())

#Apply filter to input data
yhat = model.predict(data)


#Shape of feature map output will be 4D [batch, rows, columns, filters]. 
#We did a single batch and have single filter (one filter and one input channel), so output shape is [1, ?, ?, 1].
#Can pretty-print content of the single feature map as follows:

#iterate over all rows in feature map
for r in range(yhat.shape[1]):
	#Print each column in this row
	print([yhat[0, r, c, 0] for c in range(yhat.shape[2])])

#The filter was applied top-left to bottom-right, using dot product each time