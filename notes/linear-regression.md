## Linear Regression
In statistics regression is a measure of relation between the mean value and
another value. We can think of this as the relationship between two things.

In machine learning the relationship is between independent values and the
dependant value or outcome.

As an example, the higher you climb or walk or what-ever, the colder it gets, so
the temperature depends on the height/altitude. Other factors may also effect
the temperature like humidity which is an independent variable.
In this case temperature is the dependent value, and humidity an independent
value.

We can express this as:
```
 y ~= f(x, w)

y = dependent variable (temperature)
x = independent variables (humidity, etc)
w = weights are cooefficiants of the x-term
```
There can be any number of independent variables, xs: 
```
y = w₁x₁ + w₂x₂ + w₃x₃
```
And the weights would be cooefficiants of these xs independent variables.

We would know the x values and the y value. But it is the weights that we don't
know and that we want to figure out.

If we take a look at our first tensorflow example,
[first.py](../tensor-flow/src/first.py), we provide the know x values, and the
known y values:
```python
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
```
So lets say that xs are values of humidity.
```
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
```
So we can think of this as a function:
```
 y = f(x)
```
Where we don't know what the function `f` does, but we do know what it returns
for the given x:
```
  -3.0 = f(-1.0)
  -1.0 = f( 0.0)
   1.0 = f( 1.0)
   3.0 = f( 2.0)
   5.0 = f( 3.0)
   7.0 = f( 4.0)
```
Now, notice that our function only take one argument, so we only have one x
value in the equation/function body of `f`:
```
  y = f(x) = wx + b
```
If we can figure out the correct values of `w` and `b` we can pass in any x
value and predict the "correct" value of y.

When there is only a single input value this is called
`Simple Linear Regression`.
```
y = 2x - 1
```
