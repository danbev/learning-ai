def perceptron_neuron(x, w):
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i]

    return signum(z)

def signum(s):
    if s < 0: # sign/signum function
        return -1
    else:
        return 1


# Only valid input values are 1 and -1
bias = 1
xs = [bias, -1.0, -1.0]
ws = [0.9, -0.6, -0.5]

output = perceptron_neuron(ws, xs)
print(output)

