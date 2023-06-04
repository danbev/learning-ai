import random

def perceptron_neuron(x, w):
    z = 0.0
    for i in range(len(w)):
        #print(f'w[{i}] * x[{i}] = z')
        #print(f'{w[i]} * {x[i]} = {w[i] * x[i]}')
        z += w[i] * x[i]

    return signum(z)

def signum(s):
    if s < 0: # sign/signum function
        return -1
    else:
        return 1

def sigbool(v):
    if v < 0: # sign/signum function
        return False
    else:
        return True

# Only valid input values are 1 and -1
bias = 1
ws = [0.9, -0.6, -0.5]

print(sigbool(perceptron_neuron(ws, [bias, -1.0, -1.0])))
print(sigbool(perceptron_neuron(ws, [bias, -1.0,  1.0])))
print(sigbool(perceptron_neuron(ws, [bias,  1.0, -1.0])))
print(sigbool(perceptron_neuron(ws, [bias,  1.0,  1.0])))
"""
Notice that with the above inputs this neuron acts like a NAND gate
AND          NAND
1 & 1 = 1    1 ^ 1 = 0
1 & 0 = 0    1 ^ 0 = 1
0 & 1 = 0    0 ^ 1 = 1
0 & 0 = 0    0 ^ 0 = 1

And we know that we can combine multiple NAND gates to build any logical
function.

So the weights are what allow this perceptron to act as a NAND gate. We should
therefor be able to adjust the weights to make the perceptron acts as a AND
gate. So just trying to guess is difficult which I actually tried doing, but
this what training of the network/model is all about, coming up with the
weights.
"""

"""
This part is showing how we can learn the weights for a NAND gate.
"""
LEARNING_RATE = 0.1
index_list = [0, 1, 2, 3]

# These are all the combinations of input values to a NAND gate.
#           Bias     NAND Table
x_train = [ (1.0,    -1.0, -1.0), # false ^ false  = true
            (1.0,    -1.0,  1.0), # false ^ true   = true
            (1.0,     1.0, -1.0), # true  ^ false  = true
            (1.0,     1.0,  1.0)] # true  ^ true   = false
# The expected output:
y_train = [1.0,  # -1.0 ^ -1.0 =  1.0
           1.0,  # -1.0 ^  1.0 =  1.0
           1.0,  # -1.0 ^  1.0 =  1.0
           -1.0] #  1.0 ^  1.0 = -1.0


def show_learning(w):
    print('w0 =', '%5.2f' % w[0], ', w1 =', '%5.2f' % w[1], ', w2 =', '%5.2f' % w[2])


def training():
    random.seed(7)
    ws = [random.random(), random.random(), random.random()]
    show_learning(ws)
    good_result = False;
    while not good_result:
        good_result = True
        random.shuffle(index_list)
        for i in index_list:
            x = x_train[i] # select on of the 
            y = y_train[i]
            z = perceptron_neuron(ws, x)
            if y != z:
                for j in range(0, len(ws)):
                    ws[j] += (y * LEARNING_RATE * x[j])
                    good_result = False
                    show_learning(ws)


training()

ws = [0.3, -0.5, -0.35] # taken from the output of the training in the previous step
print("Training to learn weights for a NAND gate")
print(sigbool(perceptron_neuron(ws, [bias, -1.0, -1.0])))
print(sigbool(perceptron_neuron(ws, [bias, -1.0,  1.0])))
print(sigbool(perceptron_neuron(ws, [bias,  1.0, -1.0])))
print(sigbool(perceptron_neuron(ws, [bias,  1.0,  1.0])))

"""
This part is showing how we can learn the weights for a AND gate.
"""
print("Training to learn weights for a AND gate")
# These are all the combinations of input values to a NAND gate.
#           Bias     AND Table
x_train = [ (1.0,    -1.0, -1.0), # false ^ false  = false
            (1.0,    -1.0,  1.0), # false ^ true   = false
            (1.0,     1.0, -1.0), # true  ^ false  = false
            (1.0,     1.0,  1.0)] # true  ^ true   = true
# The expected output:
y_train = [-1.0,  # -1.0 ^ -1.0 =  1.0
           -1.0,  # -1.0 ^  1.0 =  1.0
           -1.0,  # -1.0 ^  1.0 =  1.0
            1.0] #  1.0 ^  1.0 = -1.0


training()

ws = [-0.10, 0.10, 0.05] # taken from the output of the training in the previous step
print(sigbool(perceptron_neuron(ws, [bias, -1.0, -1.0])))
print(sigbool(perceptron_neuron(ws, [bias, -1.0,  1.0])))
print(sigbool(perceptron_neuron(ws, [bias,  1.0, -1.0])))
print(sigbool(perceptron_neuron(ws, [bias,  1.0,  1.0])))

