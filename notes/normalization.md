## Normalization
Normalization simplifies the problem space for machine learning algorithms,
making it easier and faster for them to learn.

Lets take a simple example to understand this:
```
array = [2, 4, 4, 4, 5, 5, 7, 9, 12, 18]
sum = 70
mean = 7
```
The standard deviation tells us how much each value deviates from the mean on
average. Note that this says each value, so we need iterator over all the
values. It also says on average, so we need to sum them up and then devide by
the number of elemements.

Let calculate this manually:
```
(2 - 7)^2 = 25
(4 - 7)^2 = 9
(4 - 7)^2 = 9
(4 - 7)^2 = 9
(5 - 7)^2 = 4
(5 - 7)^2 = 4
(7 - 7)^2 = 0
(9 - 7)^2 = 4
(12 - 7)^2 = 25
(18 - 7)^2 = 121
```
Here we are subtracting the mean from each value which will give us the distance
of that point to the mean. This value can be negative which is something we
don't want as that would cancel out the positive values. So we square the
distance to get a positive value and this also amplifies larger deviations from
the mean.

We then sum them up and devide by the number of elements:
```
sum = 25 + 9 + 9 + 9 + 4 + 4 + 0 + 4 + 25 + 121 = 210
```
And then we want the average so we divide by the number of elements. 
And then to get the standard deviation we take the square root of that undo
the squaring that we did earlier:
```
std = sqrt(210 / 10) = 4.583
```
What we have done so far is just calculated the standard deviation for the
range of values we have. That is for our array of values, on average each point
is 4.583 (deviations) away from the mean.

The standard deviation is how many standard deviations away from the mean a
value is. 

Next we center the data around zero by subtracting the mean from each value in
the array:
```
value = (value - mean) / std
```
By subtracting the mean, we are shifting the dataset so that it's centered
around zero.
Why do we want to center the data around zero?  
Optimization Algorithms: Many machine learning algorithms, especially those that
use gradient descent as an optimization technique, converge faster when the data
is centered around zero. This is because the gradients have a more consistent
scale across different features.

Regularization: Regularization techniques, which help to prevent overfitting,
often assume that all features have zero mean. Centering your data around zero
ensures that the regularization term works as expected.

Interpretability: When the data is centered, it's easier to understand the
impact of each feature on the model's output. A feature value of zero means it's
at the average level, and this provides a natural baseline for interpretation.

Dividing by the standard deviation scales the data, but the center remains at
zero.

And if we do that for all the values in our array:
```
-1.091089451179962
-0.6546536707079772
-0.6546536707079772
-0.6546536707079772
-0.4364357804719848
-0.4364357804719848
0.0
0.4364357804719848
1.091089451179962
2.400396792595916
```
Different features in a dataset can have different units and ranges, making it
difficult for machine learning algorithms to interpret them equally.
Normalization brings all variables to a similar scale, making it easier for
algorithms to understand the data.

Zero can be important to have 0.0 in the floating point range, for example when
using a sigmoid function. If we have a value that is very large, then the
sigmoid function will return 1.0. If we have a value that is very small, then
the sigmoid function will return 0.0
