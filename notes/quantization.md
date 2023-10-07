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
