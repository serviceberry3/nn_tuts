#HAND-TRAIN A NN TO ESTIMATE DIGITS THAT THE PIXELS REPRESENT

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import sys

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
#don't use ellipsis to truncate arrays when printing
np.set_printoptions(threshold=sys.maxsize)

print(X_train)

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

''' For input layer, need 64 nodes to cover 64 pixels in the image. As discussed, need 10 
output layer nodes to predict digits. Also need a hidden layer in our network to allow for complexity of the task.
Usually, num of hidden layer nodes is between num of input layers and the number of output layers. '''
nn_structure = [64,30,10] #30 nodes for hidden layer

#set up sigmoid activation function and its derivative for the backprop calculations
def f(x):
    return 1 / (1 + np.exp(-x))

def f_deriv(x):
    return f(x) * (1 - f(x))

import numpy.random as r

#start with random weights
def setup_and_init_weights(nn_structure):
    #intitialize weights and biases to empty dict
    W = {}
    b = {}

    #iterate over the three dimensions, skipping the input layer
    for l in range (1, len(nn_structure)):
        #weights: sets up key in dict as the dimension num, and the value as 2D array of random values (#nodes in this layer) x
        #(# nodes in previous layer)
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))

        #biases: sets up key in dict as dimension num, and value as 1D array of random values (length of this dimension),
        #signifying a separate bias to be added for each node
        b[l] = r.random_sample((nn_structure[l],))
        
    #see what we got
    print(W)
    print(b)
    return W, b

#set mean accumulation values ΔW and Δb to zero (need to be same size as weight and bias matrices)
def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}

    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))

    return tri_W, tri_b


#feed_forward pass through the network to calculate the 10 outputs given a set of 64 inputs
def feed_forward(x, W, b): #takes set of inputs(1D of len 64), weights dict (keys to 2D), and biases dict (keys to 1D)
    h = {1: x}
    z = {}

    #iterate from 1 thru (num of total layers - 1), which is just 2 in our case. So will iterate just twice
    for l in range(1, len(W) + 1):
        #if first layer, input into the weights is x
        if l == 1:
            node_in = x

        #otherwise, input into the weights is output from last layer
        else:
            node_in = h[l]

        #CALCULATE OUTPUT OF NEXT LAYER

        #at key (layer number + 1) in z, store result of:
        #matrix arithmetic. multiply appropriate 2D weights array (matrix) by node_in
        #matrix, then add the appropriate 1D biases matrix for this layer, to get 30x64 matrix (if had 30x64 wts
        #mat and 1x64 input) of inputs to the next layer, OR 10x64 mat (if had 10x30 wts mat and 30x64) of final outputs
        z[l+1] = W[l].dot(node_in) + b[l] #z is input to the activ fxn for a given layer
        '''Formal notation: z^(l+1) = W^(l) * h^(l) + b^(l)'''
        

        #at key (layer number + 1) in h, store result of running this z through the activation function. So we're passing
        #result as node_in of next layer
        h[l+1] = f(z[l+1]) #h is output of a given layer
        '''Formal notation: h^(l) = f(z^(l))'''

    
    return h, z

'''calculate output layer delta, which equals rate of change of cost fxn for entire newtork w.r.t. input z for this node of
output lyr. This can be expanded to (rt of change of cost fxn w.r.t. output of this output lyr node) *
(rt of chng of output of this output lyr node w.r.t. input z for this node of output lyr). That's what's shown below
after the derivatives are calculated'''
def calculate_out_layer_delta(y, h_out, z_out):
    '''Formal notation: delta_i^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))'''
    return -(y - h_out) * f_deriv(z_out)

def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    '''Formal notation: delta^(l) = (transpose(W^l)) * delta^(l+1)) * f'(z^(l))'''
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)

#train the network
def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25):
    #initialize the weights and biases
    W, b = setup_and_init_weights(nn_structure)

    cnt = 0

    #set m to number of training data we have
    m = len(y)

    avg_cost_func = []

    print('Starting gradient descent for {} iterations'.format(iter_num))

    #do iter_num iterations
    while cnt < iter_num:
        if cnt % 1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))

        #initialize cumulative sums
        tri_W, tri_b = init_tri_values(nn_structure)

        avg_cost = 0

        #iterate over all training data
        for i in range(len(y)):
            delta = {}

            #perform feed-forward pass and return stored h and z values to be used in gradient descent step
            h, z = feed_forward(X[i, :], W, b)

            #loop from nl-1 to 1 backpropogating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    #store the out layer delta if we're on output layer
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])

                    avg_cost += np.linalg.norm((y[i,:] - h[l]))

                else:
                    if l > 1:
                        #get hidden layer delta
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])

                    '''Formal notation: triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))'''
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:, np.newaxis]))

                    '''Formal notation: trib^(l) = trib^(l) + delta^(l+1)'''
                    tri_b[l] += delta[l+1]

        #perform gradient descent step for wts in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])

        #complete avg cost calc
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt+=1

    return W, b, avg_cost_func


#run a training session of the model to get ideal weights and biases
W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)

                    


    




    


        








