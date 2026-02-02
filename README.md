# Project: NumPyNet (Deep Learning from Scratch)

## 🎯 Objective
Build a modular deep learning library using **only NumPy** to demonstrate mastery of the mathematical foundations of backpropagation and gradient descent.

## 🛠 Tech Stack
- **Language:** Python 3.12
- **Math:** NumPy
- **Visuals:** Matplotlib

## 📋 Core Requirements
1. **Layer Logic:** Create an abstract `Layer` class with `.forward()` and `.backward()` methods.
2. **Dense Layer:** Implement weight initialization and gradient calculation for $W$ and $b$.
3. **Activations:** Implement `ReLU`, `Sigmoid`, and their derivatives.
4. **Loss Functions:** Implement `Mean Squared Error` and `Categorical Cross-Entropy`.
5. **Optimizer:** Implement `Stochastic Gradient Descent (SGD)`.

## 🚀 Milestones
- [ ] **XOR Verification:** Successfully train a 2-layer MLP to solve the XOR logic gate.
- [ ] **MNIST Classification:** Train a 3-layer network on the MNIST dataset (digit recognition).
- [ ] **Mini-Batching:** Implement data shuffling and batch processing for efficiency.
- [ ] **Benchmarking:** Compare training speed and accuracy against a PyTorch equivalent.

## 📈 Success Criteria
- **Math:** Zero use of autograd libraries; all derivatives manually coded.
- **Accuracy:** Solid test accuracy on MNIST digits.
- **Documentation:** Clear derivation of the chain rule used in the `backward` pass.
