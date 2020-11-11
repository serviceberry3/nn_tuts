#Example of using gradient descent for solving minimum of the
#equation f(x) = x^4-3x^3+2

#value doesn't matter as long as abs(x_new - x_old) > precision
x_old = 0

#the algo starts at x=6
x_new = 6

#step size
gamma = 0.01

precision = 0.00001

#calculate derivative (gradient) of the original fxn at x
def derivative_calc(x):
    y = 4 * x**3 - 9 * x**2
    return y


while abs(x_new - x_old) > precision:
    #set x_old to x_new from last time (initially x_old=x_new=6)
    x_old = x_new

    #calculate the derivative (slope) of the fxn at x val equal to last x_new
    #multiply that by -0.01 and add it to x_new. Once this addition is <=0.00001,
    #that means the slope is very tiny at this x value, and we've found a local min
    x_new += -gamma * derivative_calc(x_old) #note if slope is negative we go forwards to
    #find min, while if it's pos we go backwards to find min. The bigger the slope we find,
    #the bigger we'll change our guess of x value

#found the local minimum
print("The local min (closest to x=6) occurs at %f" % x_new)
