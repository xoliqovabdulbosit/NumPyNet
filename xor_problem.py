import numpy as np
from main import NeuralNetwork, Linear, Tanh, mse, mse_prime

# 1. Prepare Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# 2. Define Network Architecture
layers = [
    Linear(2, 3),
    Tanh(),
    Linear(3, 1),
    Tanh()
]
model = NeuralNetwork(layers)

# 3. Train the Model
print("Starting training...")
model.train(
    X,
    Y,
    epochs=5000,
    learning_rate=0.1,
    loss_function=mse,
    loss_prime=mse_prime
)

# 4. Final Predictions
print("\nFinal Predictions:")
for x in X:
    prediction = model.predict(x.reshape(1, -1))
    print(f"Input: {x}, Predicted: {prediction[0][0]:.4f}")
