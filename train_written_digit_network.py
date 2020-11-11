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


