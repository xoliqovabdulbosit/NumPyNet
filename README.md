
# NumPyNet: Deep Learning from Scratch

NumPyNet is a modular, object-oriented Deep Learning library built entirely from scratch using **NumPy**. By bypassing high-level frameworks like PyTorch, this project demonstrates a deep-level understanding of the mathematical foundations of neural networks, specifically backpropagation and gradient descent.

## 🌟 Key Features
- **Modular OOP Design:** Base `Layer` class architecture allows for easy extension of new layer types or activations.
- **Automated Training Pipeline:** A `NeuralNetwork` wrapper class that handles the training loop, forward/backward passes, and loss calculation.
- **Manual Gradient Calculation:** All derivatives and weight updates are manually coded without the use of `autograd`.
- **Weight Initialization:** Implements Xavier Glorot inspired initialization to prevent vanishing and exploding gradients.

## 🧠 Mathematical Foundations
The core of this library is the **Chain Rule**. Each layer is responsible for:
1.  **Forward Pass:** $Y = f(X, W, b)$ — Computing the output and caching the input.
2.  **Backward Pass:** $\frac{\partial E}{\partial X} = \frac{\partial E}{\partial Y} \cdot \frac{\partial Y}{\partial X}$ — Computing the gradient of the loss with respect to both parameters and inputs.

## 🛠 Tech Stack
- **Language:** Python 3.12
- **Core Math:** NumPy
- **Standard:** PEP 8 Compliant OOP

## 🚀 Quick Start: Solving XOR
The XOR problem is the "Hello World" of Deep Learning, as it requires a non-linear decision boundary.

```python
from numpynet.layers import Linear, Tanh
from numpynet.model import NeuralNetwork
from numpynet.losses import mse, mse_prime

# 1. Define Architecture
model = NeuralNetwork([
    Linear(2, 3),
    Tanh(),
    Linear(3, 1),
    Tanh()
])

# 2. Train
model.train(X, Y, epochs=5000, learning_rate=0.1, loss_func=mse, loss_prime=mse_prime)

# 3. Predict
predictions = model.predict(X)
```
