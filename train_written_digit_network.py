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
X=X_scale.fit_transform(digits.data)

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




