## Reinforcement learning
This is a type of machine learning where the model is trained by interacting
with the environment. The model is rewarded for correct actions and penalized
for incorrect actions. The model is trained by trial and error and the goal is
to maximize the reward. The model is not given any examples of correct actions
and is not told what to do, it has to figure it out by itself.

If we want to train a model to play the game of pong we would first record the
images from a game played by a human. Yes, that is correct, it is trained on
images/pixels from game played. I thought this was very facinating and not what
I expected (not that I had any expectations but still surprised me).
```
      pixels
+-----------------+
|  0           1  |
|-----------------|
|                 |
| |               |        +----------------+
|           *     | -----> | Neural Network | -----> action (up or down)
|              |  |        +----------------+
|                 |
+-----------------+
```
So the pixels for a given state/frame is the input to the neural network. The
output if the prediction if the paddle should move up or down. For example, then
value 1 if the paddle should move up and -1 if the paddle should move down, and
0 otherwise (the ball is in the middle of the paddle).

These images would then be the input into an LLM which would predict the next
action to take, that is to move the paddle up or down. So the input images is
the data set that the model is trained on and is something that has to be
gathered before the training process can start.

If we want something to be able to learn "any" task then it would be difficult
to gather all the data needed to train the model, like how do we train it on
every thing. Also the llm will not be better than the human that played the
game. This was an example of supervised learning which is not what we want to
do here. What we want is an "agent" that can learn to play the game by itself
and this is where reinforcement learning comes in.

So in this case we don't have an input data set to train on. We still have the
input in the form of images. But we don't have the correct, called target
lables, for the game which in the example above was provided by the human
playing the game and saving the images. So we don't know what the correct action
is for each image. So we need to figure out a way to train the model without
having the correct labels. In RL what decides the action is called the Policy
Network.

### Policy Network
The policy network is a neural network that takes the input and predicts the
next action to take. The policy network is trained by interacting with the
environment and is rewarded for correct actions and penalized for incorrect
actions. The goal is to maximize the reward.

### Policy Gradient
The policy gradient is the gradient of the policy network.

The initial state of this network is random. We take one frame from the image
and pass it into the policy network.
```
            pixels
      +-----------------+
      |  0           1  |
      |-----------------|
+---> |                 |
|     | |               |        +----------------+
|     |           *     | -----> | Policy Network | -----> action (up or down)
| +-> |              |  |        +----------------+               |
| |   |                 |                                         |
| |   +-----------------+                                         |
| |       Reward                 +----------------+               |
| +----------------------------- | Environment    | <-------------+
+------------------------------- |                |
          State                  +----------------+
```
The reward in this case if the score of the game. If the score is highter for
the player then the reward is positive and if the score is lower then the
reward is negative.
Also the actual output of the action can be somewhat influenced so that it not
always just takes the prediction from the Policy Network but me be influenced 
by a procentage of the previous action. This is called the exploration rate.

The policy gradient is used to update the weights of the policy network. The
policy gradient is calculated by taking the gradient of the log of the policy
network output multiplied by the reward. The policy gradient is then used to
update the weights of the policy network.

### Credit Assignment Problem
The credit assignment problem is the problem of assigning credit to the actions
that lead to the reward. This is a problem because the reward is only given at
the end of the game and not for each action. For example, in pong we might
loose the game if the ball passes past us but we might have been doing really
well up until that point. It is difficult to know which action caused us to
loose the game. This is called the credit assignment problem.
The issue is that we only get a reward after a game is played which is called
sparse rewards. The agent needs to figure out what part of the all the sequences
of interaction caused the negative result.
And note that we don't really care about what happens in the game right before
the ball hits the paddle (with regards to the reward that is). This where reward
shaping comes in.

### Reward Shaping
Reward shaping is the process of giving rewards for actions that are not
directly related to the reward. For example, in pong we might give a reward for
each time the ball hits the paddle. This is called dense rewards. This is done
to help the agent learn faster. This is called reward shaping. But this has an
issue what it is specific to the specific game, which is pong in our case. So
this does not scale.


