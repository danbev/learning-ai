## Direct Preference Optimization (DPO)
The data set used in these cases have three parts, the prompt itself, and then
a rejected response and an accepted response.

When a llm is trained the first step is to gather a large data set consisting
of text, code, books, etc. Scraping the web is a common way to gather this data
and in the process this will "pick up" information that might not be suitable
for a company or organization to use. And there might be some information that
is just incorrect which we might not want either. This data set is then used to
train a model (which consists of weights and biases which is the output of the
training).
But with this model it might output things that we don't want it to, like how
to build a bomb, or how to make drugs, etc. 

To get the model to output what we want we can prompt it, for example we could
prompt it with "What is the best ice cream ?". And if we do this twice might get
the following two responses:
```
What is the best ice cream: GB
What is the best ice cream: Haägen-Dazs
```
We (humans) can then specify which response we prefered and which we did not.
For example we might prefer the second response and reject the first.
```
What is the best ice cream: GB              Rejected
What is the best ice cream: Haägen-Dazs     Preferred
```
And we can continue doing this for different things we want the model to output.
So humans prompt and pick the preferred responses.

So for a prompt x, the model generates a pair of answers (y₁, y₂) where each
answer is chosen based on the models policy πˢᶠᵗ(y | x), which dictates how
likely and possible answer is to be selected. This can be written as:
```
prompt x (y₁, y₂) ~ πˢᶠᵗ(y | x) 

y₁          = rejected response
y₂          = accepted response
~           = is distributed as, or drawn from
πˢᶠᵗ(y | x) = the policy that the model uses to generate the response y
              given the prompt x
```
This can be read as, the model generates a pair of answers (y₁, y₂) for a given
prompt x, where each answer is chosen based on the models policy πˢᶠᵗ(y | x),
which dictates how likely and possible answer is to be selected.


And these pairs of prompts and responses (winning/losing) are then presented to
humans that will pick the preferred response. This can be written as:
```
y_w ≻ y_l | x

y_w = winning/preferred response
y_l = losing/rejected response
x   = prompt
≻   = preferred over (not greater than, more like a duck mouth bent upwards)
```

All these prompts and their winning/losing responses are then collected into a
data set D:
```
D = { (xⁱ, y_wⁱ, y_lⁱ) }ⁿᵢ₌₁
```
Where D is a dataset of (x, y_w, y_l) triplets, or pair of prompt and responses
and N is the total number of items in the dataset.
```

`r^*(y₁, y₂)` represents the preferences of some latent reward model, meaning
that we can't observe it, and could be a human clicking or entering values.
One concrete way of writing this is using the Bradley-Terry (BT) model which
states that human preference `p^*` can be written as.

`p^*` referres to that the dataset D is assumed to to be sampled to the true
, but unknown, preference distribution `p^*`. Which could be human preferences
for example.
```
                    exp(r^*(x, y₁))
p^*(y₁ ≻ y₂ | x) = ----------------------------------
                    exp(r^*(x, y₁)) + exp(r^*(x, y₂))

Where `σ` is the sigmoid function and `r^*` is the latent reward model. This
```
Recall that [exp](exp.md) always returns a positive value and it also amplifies
differences between numbers. For example:
```
r^*(x, y₁) = 2
r^*(x, y₂) = 1
```

Then we can plug in those values into the BT model:
```
                    exp(2)
p^*(y₁ ≻ y₂ | x) = ----------------
                    exp(2) + exp(1)

                      7.39
                 = ---------------
                    7.39 + 2.71

                      7.39
                 = ---------------
                      10.11

                 = 0.73
```
This says that the probability of y₁ being preferred over y₂ given x is 0.73 or
73%. So notice that this is basically first making sure that the input values
are positive by using exp, and exp also amplifies the difference between them
y₁ = 2, exp(2) = 7.39 and y₂ = 1, exp(1) = 2.71. Notice that the original
difference is 2-1=1, but after using exp the difference is 7.39-2.71=4.68. So
the difference is amplified. And then we are calculating the percentage
(part/whole, the part being 7.39 and the whole being 7.39+2.71).

Next we parameterize a reward model which we write like this:
```
rφ(x, y)

rφ = reward model.
φ  = parameters of the reward model, like weights and biases for example. (phi)
```
The goal is to estimate estimate these parameters (phi) in such a way that our
model's predictions align with the observed preferences in your dataset (D).
One way of doing this is to use maximum likelihood estimation (MLE) which is
a statistical method for estimating the model parameters that make the observed
data most probable.

log-likehood loss:
```
L_R(rφ, D) =

L_R = log-likelihood loss function
rφ  = reward model
D    = dataset
```
The loss of the reward model (rφ) with respect to the dataset (D).

Negative log-likelihood loss:
```
L_R(rφ, D) = -E(x, y_w, y_l)∼D [log σ(y_w ≻ y_l | x)]

-E(x, y_w, y_l)∼D = The expected value (E) over the data set consists of
                    triplets (prompt, winning response) sampled from the data
                    set D (~D).
```
The brackets are used to clearly denote that the expected value is calculated
over the logarithm of the probabilities that represent human preferences between
two responses given a prompt. 
The expected value (E) inherently involved averaging over the data set D without
explicitly stating the division by the number of items in the data set.
Basically we can think of E as something like the following:
```
N = items in D. And set to 3 in this case.
             
              (log σ(y_w ≻ y_l | x₁)) + (log σ(y_w ≻ y_l | x₂)) + (log σ(y_w ≻ y_l | x₃))
L_R(rφ, D) =- ----------------------------------------------------------------------------------
                                                     N
```
So just to clarify the expected value here, lets look at a concrete example with
three items in the dataset D:
```
x₁ = "What is the best ice cream?"
y_w₁ = "Haägen-Dazs"
y_l₁ = "GB"

x₂ = "What is a healthy breakfast"
y_w₂ = "Oatmeal"
y_l₂ = "Bacon"

x₃ = "Which is the best editor?"
y_w₃ = "Vi"
y_l₃ = "Emacs"
```
So we start by computing the log probability log σ(rφ(x, y_w) - rφ(x, y_l)) for
each item in the data set:
```
x₁: (rφ(x₁, y_w₁) - rφ(x₁, y_l₁)) = 0.8
x₂: (rφ(x₂, y_w₂) - rφ(x₂, y_l₂)) = 0.5
x₃: (rφ(x₃, y_w₃) - rφ(x₃, y_l₃)) = 0.9

x₁: σ(0.8) = 0.69
x₂: σ(0.5) = 0.62
x₃: σ(0.9) = 0.71

x₁: log(0.69) = -0.37
x₂: log(0.62) = -0.48
x₃: log(0.71) = -0.34

    (-0.37 - 0.48 - 0.34)
E = --------------------- = -0.40
              3
```
The value of `-40` represents the expected value of the log probability of the
the model correctly identifying the preferred response over the rejected
response accross all the three items in the dataset. 

So when llm does when it predicts the next token. From the data the model was
trained on, it will predict the next token. So if during training is saw the
sentence "What is the best ice cream: Haägen-Dazs". And if it sees this many
times in the training set it will most probably predict "Haägen-Dazs" as the
next token. So it is about the frequence of the tokens in the training set. To
get the model to predict "Haägen-Dazs" as the next token, the training set must
have had many examples of "What is the best ice cream: Haägen-Dazs" or similar.

With DPO we instead of adding more sample to the training set to get the result
we want, DPO has training examples as triplets. The prompt, the rejected, and
the accepted:
```
Prompt: What is the best ice cream [ ]
Rejected: GB
Accepted: Haägen-Dazs
```
In DPO we will penalize the model for predicting "GB" and reward it for
predicting "Haägen-Dazs".

So we start with a base model, and make a copy of it. One will be the model we
update and the other will be used as a reference.
