## Quanitization
This are ways to take floating point values and turn them into ints, or I guess
the main point is that we want to make something that take more storage space,
or memory space and store it with less bits. So if we think about floating
points we have half presicion 16-bits, single precision 32-bits, and double
precision 64-bits. My understanding that models are trained using
single-precision 32-bit floating points. But these can then be quantanised to
smaller sizes. This might make the model less accurate but I'm not clear on
that yet.

My current understanding is that the models could be trained on single-precision
and then quantanised. So this will make the model size smaller, taking less
room on dist, and also less memory. But at inference time I'm not sure yet if
the the values are then converted back to single-precision or if they are
inferred using the quantanised values. Hopefully this document will sort that
out.

### Symmetric Quantanisation
Take the following example:
```
   0.0                                                             1000.0
    +--------------------------------------------------------------+
    |                                                              |
    |                                                              |
    ↓                                                              ↓
    0                                                              255
```
In this case we are taking a range of floating point values in the range from
0.0 to 1000.0 and converting them into integers in the range from 0 to 255.
We can scale the floating point values using the following formula:
```
        (f_max - f_min)
scale = ---------------
        (q_max - q_min)

         1000.0 - 0.0
scale =  ------------- = 3.9215686274509803
         255 - 0
```
So 3.9215686274509803 is the scale factor. So to convert a floating point in
our range we can use this scaling factor:
```
int quantanised_value = value / scale

float value = 500.0
500.0 / 3.9215686274509803 = 127.5
                           = 128 (rounded)
    
0.0 / 3.9215686274509803 = 0.0
```
Notice that 0.0 is symmetric to the 0 value in the quantanised range. I believe
this is why it is called symmetric quantanisation.

### Asymmetric Quantanisation
Take the following example:
```
 -40.0                                                             1000.0
    +--------------------------------------------------------------+
    |                                                              |
    |                                                              |
    ↓                                                              ↓
    0                                                              255
```
Notice that our floating point range has changed in this example. Like before
we can calculate the scale factor:
```
         1000.0 - (-40.0)     1040.0
scale =  ---------------- =   ------ = 4.07843137254902
         255 - 0              255
```
So if we quantize a value in that range, for example -40.0 then we should get
0:
```
-40.0 / 4.07843137254902 = -9.80392156862745
                         = -10 (rounded)
```
Notice that this does not lined up to 0 which is what have have above. This is
an issue and something that we need to take into account.
```
1000.0 / 4.07843137254902 = 245.09803921568627
                          = 245 (rounded)
```
Notice that this value does not line up with 255. What we can do is take
the value that we got when we converted -40.0 and add the `negative` of that
value to all the values that we convert to fix this issue in mis-alignment. 
```
-40.0 / 4.07843137254902 + -(-10) = -9.80392156862745 + 10
                                   = 0.19607843137254902
                                   = 0 (rounded)

1000.0 / 4.07843137254902 + -(-10)  = 245.09803921568627 + 10
                                    = 255.09803921568627
                                    = 255 (rounded)
```
The value that we add to all the values is called the `zero-point`. 

In the examples so far we have been using 8-bit integers and they have been in
unsigned so there range was 0-255. We could also use signed integers:
```
 -40.0                                                             1000.0
    +--------------------------------------------------------------+
    |                                                              |
    |                                                              |
    ↓                                                              ↓
  -128                                                             127
```
So in this case instead of -40.0 lining up with 0 it lines up with -128, and
1000.0 lines up with 127.

```
         1000.0 - (-40.0)     1040.0
scale =  ---------------- =   ------ = -4.07843137254902
         -128 - 127            -255 


-40.0 / -4.07843137254902 = 9.80392156862745
                          = 10 (rounded)
                         
zero-point = 10
zero-point = 10 + (-128)
zero-point = 10 -128
zero-point = -118
```
So the zero-point is -118. So we can convert a floating point value to a
quantanised value using the following formula:
```
0.0 / -4.07843137254902 + -(-118) = 0.0 + 118
                                  = 118 (rounded)

1000.0 / -4.07843137254902 + -(-118) = -245.09803921568627 + 118
                                      = -127.09803921568627
                                      = -127 (rounded)
```
