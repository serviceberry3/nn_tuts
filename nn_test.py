import numpy as np
import timeit

#fill up the layer 1 wt array with example weights
w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])

#define w2 as 2D array
w2 = np.zeros((1,3))

#get row 0, all columns of w2, set to this
w2[0,:] = np.array([0.5, 0.5, 0.5])
#print(w2)

#set up dummy values in layer 1 and 2 bias weights
b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])

#declare separate Python fxn for activation fxn
def f(x):
    return 1 / (1 + np.exp(-x)) #sigmoid fxn

#simple way of calculating output of nn
def simple_looped_nn_calc(n_layers, x, w, b): #num layers in network, x(inputs array), w(wts array), b(biases array)
    for l in range(n_layers-1): #iterate over number passed layers - 1
        #set up input array which weights will be multiplied by
        
        #if first layer, input array will be x input vector
        if l == 0:
            node_in = x
        #else input to next layer is output of previous layer
        else:
            node_in = h #h not even declared yet! Awesome!

        #set up 1D output array for nodes in layer l + 1. initialized to size (# rows in this wts array)
        h = np.zeros((w[l].shape[0],))

        #loop through rows of this weights array
        for i in range(w[l].shape[0]):
            #set up sum inside the activation fxn
            f_sum = 0

            #loop through columns of weight array
            for j in range(w[l].shape[1]):
                #multiply appropriate weight by input and add to cum sum
                f_sum += w[l][i][j] * node_in[j]

            #add this bias to the cum sum
            f_sum += b[l][i]

            #use activation fxn to calculate this output (i.e. h1, h2, h3)
            h[i] = f(f_sum)
            
    #return output array from final iteration
    return h


#set up a test
w = [w1, w2]
print("WEIGHTS:", w)
b = [b1, b2]

#a dummy x input vector
x = [1.5, 2.0, 3.0]

print("Result for naive feed fwd:", simple_looped_nn_calc(3, x, w, b))


#A faster implementation of feed-forward using matrices/linear algebra
def matrix_feed_forward_calc(n_layers, x, w, b):
    #iterate over number of layers-1
    for l in range(n_layers-1):
        #if this is first layer, set node_in to the input
        if l == 0:
            node_in = x

        #otherwise this input is output from last time
        else:
            node_in = h

        #matrix arithmetic. multiply appropriate weights matrix by 3x1 node_in
        #matrix, then add the appropriate biases matrix, to get 3x1 matrix (if had 3x3 wts
        #mat) or 1x1 mat (if had just 3x1 wts mat) of inputs to the next layer
        z = w[l].dot(node_in) + b[l]

        #finally run calculated 3x1 input matrix for this layer through the sigmoid fxn
        #to get output of this layer
        h = f(z) #yields single value if this is the last hidden layer

    #return the final output sum
    return h

print("Result for feed fwd w/matrices:", matrix_feed_forward_calc(3, x, w, b))

            
            
            
            
        
