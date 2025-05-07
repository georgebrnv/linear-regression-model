import math
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

class ManualLinearRegression:
    def __init__(self):
        self.weight = np.random.randn(1)
        self.bias = np.random.randn(1)

    def forward(self, x):
        return self.weight * x + self.bias
    

def mean(values):
    total = 0
    for value in values:
        total += value

    return total / len(values)


def loss_fn(predictions, true_values):
    differences = []
    for pred, true in zip(predictions, true_values):
        diff = abs(pred - true)
        differences.append(diff)

    return mean(differences)


def calculate_gradient(predictions, true_values, X):
    diff_sums_weight = 0
    diff_sums_bias = 0
    for pred, true, x in zip(predictions, true_values, X):
        diff = pred - true
        sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
        diff_sums_weight += sign * x
        diff_sums_bias += sign

    weight_grad = (1/len(predictions)) * diff_sums_weight
    bias_grad = (1/len(predictions)) * diff_sums_bias

    return weight_grad, bias_grad


def update_parameters(weight, weight_grad, bias, bias_grad, lr):
    updated_weight = weight - (lr * weight_grad)
    updated_bias = bias - (lr * bias_grad)

    return updated_weight, updated_bias

    
# Initialize data
weight = 0.44
bias = 1.2
X = np.arange(0, 1, 0.01).reshape(-1, 1)
y = weight * X + bias
train_split = int(len(X) * 0.8)

X_train, y_train = X[:train_split], y[:train_split] 
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        predictions=None,
    ):

    plt.figure(figsize=(10, 7))
    plt.scatter(X_train, y_train, c="b", s=5, label="Training data")
    plt.scatter(X_test, y_test, c="g", s=5, label="Test data")
    
    if predictions is not None:
        plt.scatter(X_train, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


model_0 = ManualLinearRegression()
epochs = 1351

for epoch in range(epochs):
    predictions = model_0.forward(X_train)

    # Calculate loss for predictions
    loss = loss_fn(predictions=predictions, true_values=y_train)

    # Calculate gradients
    weight_grad, bias_grad = calculate_gradient(predictions=predictions, true_values=y_train, X=X_train)

    # Update weight and bias
    if epoch > 1000:
        updated_weight, updated_bias = update_parameters(model_0.weight, weight_grad, model_0.bias, bias_grad, lr=0.00001)
    elif epoch > 700:
        updated_weight, updated_bias = update_parameters(model_0.weight, weight_grad, model_0.bias, bias_grad, lr=0.0001)
    elif epoch > 500:
        updated_weight, updated_bias = update_parameters(model_0.weight, weight_grad, model_0.bias, bias_grad, lr=0.001)
    else:
        updated_weight, updated_bias = update_parameters(model_0.weight, weight_grad, model_0.bias, bias_grad, lr=0.01)
    model_0.weight = updated_weight
    model_0.bias = updated_bias

    if epoch % 150 == 0 or epoch == 0:
        print(f"""
                                EPOCH: {epoch}

        Loss: {loss} | Weight: {model_0.weight} | Bias: {model_0.bias}
        Gradients - Weight grad: {weight_grad} | Bias grad: {bias_grad}
        Updated params - Upd Weight: {updated_weight} | Upd Bias {updated_bias}
    """)
        plot_predictions(predictions=predictions) # Random weight and bias
