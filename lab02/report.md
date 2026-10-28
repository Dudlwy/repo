1.Goal

Design and implement a two-hidden-layer neural network from scratch using NumPy

2.Method

We took a set of thousands of tiny pictures of handwritten numbers, cleaned up the data, and built a custom computer brain with two hidden layers using basic math rules. We taught the computer how to guess the numbers, double-checked our mathematical equations by slightly tweaking the numbers to make sure our code had no errors, trained the network over fifty rounds with different setup speeds and sizes, and measured how fast the computer made guesses compared to how fast it learned from its mistakes.

3.Results table or plot

![alt text](image.png)

Three conclusions in your own words

Analytical gradient checking achieved relative error well below 1e-6 across all sampled parameters


The backward pass took approximately twice as long as the forward pass

Increasing hidden layer width from 16 to 64 along with an optimal learning rate of 0.1 improved test accuracy