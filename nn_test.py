import numpy as np
import timeit

#fill up the layer 1 wt array with example weights
w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])

#define w2 as 2D array
w2 = np.zeros((1,3))

#get row 0, all columns of w2, set to this
w2[0,:] = np.array([0.5, 0.5, 0.5])

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
b = [b1, b2]

#a dummy x input vector
x = [1.5, 2.0, 3.0]

print(simple_looped_nn_calc(3, x, w, b))

            
            
            
            
        
