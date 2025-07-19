import matplotlib.pyplot as plt
import torch
from torch import nn

torch.manual_seed(4)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.weight * x + self.bias
    

def plot_predictions(X_train, X_test, y_train, y_test, predictions=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, s=5, c='g', label='Training Data')
    plt.scatter(X_test, y_test, s=5, c='r', label='Test Data')

    if predictions is not None:
        plt.scatter(X_test, predictions, s=5, c='y', label='Predictions')
    
    plt.legend(prop={'size': 10})
    plt.show()


def training(model: nn.Module,
             epochs: int,
             X_train: torch.Tensor,
             y_train: torch.Tensor,
             X_test: torch.Tensor,
             y_test: torch.Tensor,
             loss_fn: nn.Module,
             optimizer: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler._LRScheduler):
    try:
        model.train()
        for epoch in range(epochs):
            y_predictions = model.forward(X_train)
            loss = loss_fn(y_predictions, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch} --> Train Loss: {loss}')
                testing(model, X_test, y_test, loss_fn, epoch)

    except Exception as e:
        raise Exception(f"An error occurred during TRAINING: {e}")

def testing(model: nn.Module,
            X_test: torch.Tensor,
            y_test: torch.Tensor,
            loss_fn: nn.Module,
            epoch: int,
            ):
    try:
        model.eval()
        with torch.inference_mode():
            y_pred = model.forward(X_test)
            loss = loss_fn(y_pred, y_test)

            print(f'           Test Loss: {loss}')

        if (epoch + 1) % 100 == 0:
            plot_predictions(X_train, X_test, y_train, y_test, y_pred)
    except Exception as e:
        raise Exception(f"An error occurred during TESTING: {e}")
    

# Determine weight and bias for linear function that needs to be guessed by model
weight = 0.34
bias = 1.3

# Prepare dataset for suprivized learning
start = 0
end = 1
step = 0.025

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(len(X) * 0.8)

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train))
print(len(X_test), len(y_test))

# Create a model instance
linear_model = LinearRegressionModel()

# Define loss function and oprimizer
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.SGD(params=linear_model.parameters(), lr=0.01)

# Add scheduler for a dynamic learning rate
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    step_size=500,
    gamma=0.009
)

# Run
epochs = 1000
training(linear_model, epochs, X_train, y_train, X_test, y_test, loss_fn, optimizer, scheduler)
