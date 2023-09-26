import numpy as np

# This is just the same example from the notes:
floats = [4.8, 5.0, 5.2, 6.0, 7.0]
print(f'{floats=}\n')

# The following is how we can convert the floats to signed integers (int8)
a = 0.1
offset = (4.8 + 7.0)/2
print(f'{offset=}\n')

print("Converting to int8 (signed ints):")
ints = []
for f in floats:
    ints.append(int((f - offset) / a))

print(f'{ints=}\n')

# And we should be able to convert back to floats:
print("Converting int8's back to floats:")
back_to_floats = []
for i in ints:
    print(f'{i} * {a} + {offset} = {i * a + offset}')
    back_to_floats.append(i * a + offset)   
print(f'{back_to_floats=}\n')

# We can also do this for unsigned integers (uint8)
offset = 4.8 # here we choose to set the offset to the minimum value float value
# This means that the smallest float value will be mapped to 0, for example
# in our case 4.8 is the smalles value, and if we take it and subtract the offset
# we get 0.0 and then we normalize by 255 which give 0.0/255 = 0.0.
a = (7.0 - offset) / 255 
print(f'{offset=}\n')
print(f'{a=}\n')

print("Converting to int8 (signed ints):")
uints = []
for f in floats:
    uints.append(int((f - offset) / a))

print(f'{uints=}\n')

print("Converting unt8's back to floats:")
back_to_floats = []
for i in uints:
    print(f'{i} * {a} + {offset} = {i * a + offset}')
    back_to_floats.append(i * a + offset)   
print(f'{back_to_floats=}\n')

dataset = np.random.normal(size=(2, 10))
print(f'{dataset=}\n')

# Gather the min and max values for each dimension
ranges = np.vstack((np.min(dataset, axis=0), np.max(dataset, axis=0)))
print(f'mins values: {ranges[0]=}\n')
print(f'maxs values: {ranges[1]=}\n')

starts = ranges[0,:]

# For each values in the ranges, the min and max values we normalize them
# by dividing by 255.
#        max values    min values
steps = (ranges[1,:] - ranges[0,:]) / 255 # 8 bit quantization, 2^8 = 256 

print(f'({ranges[1][0]=} - {ranges[0][0]=}) = {(ranges[1][0] - ranges[0][0])/255}\n')
print(f'{steps=}\n')
print(f'{dataset[0]=}\n')

dataset_quantized = np.uint8((dataset - starts) / steps)
print(f'{dataset_quantized=}\n')
