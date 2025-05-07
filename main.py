import torch
from torch import nn
import matplotlib.pyplot as plt


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias
    
def plot_predictions(
        X_train,
        y_train,
        X_test,
        y_test,
        predictions=None
    ):

    plt.figure(figsize=(10, 7))
    plt.scatter(X_train, y_train, c="b", s=4, label="Training data")
    plt.scatter(X_test, y_test, c="g", s=4, label="Test data")

    if predictions is not None:
        plt.scatter(X_test, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={'size': 14})
    plt.show() 

def training(
        model: nn.Module,
        epochs: int,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        scheduler: torch.optim.lr_scheduler._LRScheduler
    ):
    model.train()
    for epoch in range(epochs):
        y_preds = model(X_train)
        loss = loss_fn(y_preds, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}]")
            testing(model, X_test, loss_fn, y_test)

def testing(
        model: nn.Module,
        X_test: torch.Tensor,
        loss_fn: nn.Module,
        y_test: torch.Tensor,
    ):
    model.eval()
    with torch.inference_mode():
        y_preds = model(X_test)
        loss = loss_fn(y_preds, y_test)
    
    print(f"\n\nTest Loss: {loss}")
    print(model.state_dict())


weight = 0.44
bias = 1.2

start = 0
end = 2
step = 0.02

torch.manual_seed(55)
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
train_split = int(len(X) * 0.8)

X_train, y_train = X[:train_split], y[:train_split] 
X_test, y_test = X[train_split:], y[train_split:]

model_0 = LinearRegressionModel()

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(
    params=model_0.parameters(),
    lr=0.01,
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    step_size=650, 
    gamma=0.01
)
epochs = 1501

training(model_0, epochs, loss_fn, optimizer, X_train, y_train, X_test, y_test, scheduler)

# After training, get final predictions and plot
model_0.eval()
with torch.inference_mode():
    final_preds = model_0(X_test)
plot_predictions(X_train, y_train, X_test, y_test, predictions=final_preds)
