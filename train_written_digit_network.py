#HAND-TRAIN A NN TO ESTIMATE DIGITS THAT THE PIXELS REPRESENT

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

#load up the digit-pixel map dataset
digits = load_digits()

#log shape of input data matrix
print(digits.data.shape)

#make sure we can get the images, test by displaying the "1" image
plt.gray()
plt.matshow(digits.images[1])
plt.show()

#let's look at one of the dataset pixel representations. Display row 0, all columns
#of digits.data
print(digits.data[0,:])

'''It’s common practice to scale input data so it all fits mostly between
either 0 to 1 or with a small range centered around 0 (like -1 to 1). This can help
convergence of the neural network and is imp if we're combining diff data types.'''

from sklearn.preprocessing import StandardScaler

#instantiate a scaler object
X_scale = StandardScaler()

#scale the data to fit between -2 and 2 (subtract mean and div by stdev)
X = X_scale.fit_transform(digits.data)

#make sure data was scaled
print(X[0,:])

'''To make sure that we're not creating models that are too complex ("overfitted"), it is
common practice to split dataset into training set and test set. Training set is data that
model will be trained on; test set is data that model will be tested on after it's been trained.
Amt of training data is always more numerous than testing data and usually between 60-80%
of the total dataset.'''
from sklearn.model_selection import train_test_split

#set y to the "correct digit" data
y = digits.target

#randomly separate training and testing sets: make training 60% of data, testing 40%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

'''Need output layer to predict whether digit represented by input pixels is between 0 and 9. Sensible nn architecture for this
would be to have output layer of 10 nodes, each representing a digit from 0-9. Want to train network so that when image of the
digit “5” is presented, node in output layer representing 5 has highest value. Ideally, output like this:
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]. But can settle for something like:
[0.01, 0.1, 0.2, 0.05, 0.3, 0.8, 0.4, 0.03, 0.25, 0.02] and take max index of output array and call that our predicted digit.
For MNIST data supplied in scikit learn dataset, the “targets” or classification of the handwritten digits is in form of
single number. Need to convert that num into vector so it lines up with our 10 node output layer. In other words, if
target val in dataset is “1”, want to convert it into: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0].'''
import numpy as np

#converts 1D array of correct digits to a 2D array with digits replaced as 1D arrays as shown in above blurb
def convert_y_to_vect(y):
    #intialize output vector to 2D array of zeroes
    y_vect = np.zeros((len(y), 10))

    #iterate over all digits
    for i in range(len(y)):
        #set the appropriate slot in this digit's vector to 1
        y_vect[i, y[i]] = 1

    return y_vect

#convert training and test y datasets to vector representations of the digits
y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)


#verify that the conversion worked
print(y_train[0], y_v_train[0])






