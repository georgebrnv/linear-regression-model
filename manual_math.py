import matplotlib.pyplot as plt
import numpy as np


class LinearRegressionModel():
    def __init__(self):
        self.weight = np.random.rand(1)
        self.bias = np.random.rand(1)

    def forward(self, x):
        return self.weight * x + self.bias
    

def plot_predictions(
        X_train,
        y_train,
        X_test,
        y_test,
        predictions=None,
    ):

    plt.figure(figsize=(10, 7))
    plt.scatter(X_train, y_train, c="b", s=5, label="Training data")
    plt.scatter(X_test, y_test, c="y", s=5, label="Test data")
    
    if predictions is not None:
        plt.scatter(X_test, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()
    

def loss_fn(predictions, true_values):
    diff_sum = 0
    for prediction, true_value in zip(predictions, true_values):
        diff = abs(prediction - true_value)
        diff_sum += diff
    
    return (1/len(predictions)) * diff_sum
    
def calculate_gradient(predictions, true_values, X):
    sum_diff_weight = 0
    sum_diff_bias = 0
    for prediction, true_value, x in zip(predictions, true_values, X):
        diff = prediction - true_value
        sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
        sum_diff_weight += sign * x
        sum_diff_bias += sign

    n = len(predictions)
    weight_grad = (1/n) * sum_diff_weight
    bias_grad = (1/n) * sum_diff_bias

    return weight_grad, bias_grad

def update_parameters(weight, weight_grad, bias, bias_grad, lr):
    updated_weight = weight - (lr * weight_grad)
    updated_bias = bias - (lr * bias_grad)

    return updated_weight, updated_bias

def training(model: LinearRegressionModel, epochs: int, X_train, y_train, X_test, y_test):
    for epoch in range(epochs):
        y_pred = model.forward(X_train)
        loss = loss_fn(y_pred, y_train)

        weight_grad, bias_grad = calculate_gradient(y_pred, y_train, X_train)
        updated_weight, updated_bias = update_parameters(model.weight, weight_grad, model.bias, bias_grad, lr=0.01)
        model.weight = updated_weight
        model.bias = updated_bias

        if (epoch + 1) % 100 == 0:
            print(f"""
                                EPOCH: {epoch}

        Loss: {loss} | Weight: {model.weight} | Bias: {model.bias}
        Gradients - Weight grad: {weight_grad} | Bias grad: {bias_grad}
        Updated params - Upd Weight: {updated_weight} | Upd Bias {updated_bias}
    """)
            # Compute predictions on test data for plotting
            test_predictions = model.forward(X_test)
            plot_predictions(X_train, y_train, X_test, y_test, test_predictions)

# Prepare dataset
weight = 1.1
bias = -0.3

start = 0
end = 2
step = 0.025

X = np.arange(start, end, step).reshape(-1, 1)
y = weight * X + bias

train_split = int(len(X) * 0.8)

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

model = LinearRegressionModel()
epochs = 1000

training(model, epochs, X_train, y_train, X_test, y_test)

final_predictions = model.forward(X_test)  # Use X_train instead of X_test
plot_predictions(X_train, y_train, X_test, y_test, final_predictions)