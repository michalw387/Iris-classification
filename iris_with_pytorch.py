import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    def __init__(self, input_num=4, h1=128, h2=64, output_num=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_num, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_num),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


iris = datasets.load_iris()
data = torch.tensor(iris.data, dtype=torch.float32)
data = (data - data.mean(dim=0)) / data.std(dim=0)
labels = torch.tensor(iris.target, dtype=torch.long)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)


model = Model().to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 500
losses = []

for epoch in range(epochs):
    y_pred = model(x_train.cuda())
    loss = criterion(y_pred, y_train.cuda())
    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

predicted_test = model(x_test.cuda())
predicted_test = nn.Softmax(dim=1)(predicted_test)
predicted_test = torch.max(predicted_test, 1)[1]

accuracy_test = accuracy_score(y_test, predicted_test.cpu().numpy())
f1_test = f1_score(y_test, predicted_test.cpu().numpy(), average="macro")

print("Test Accuracy:", accuracy_test)
print("Test F1 Score:", f1_test)

plt.figure(figsize=(8, 6))
plt.plot(range(epochs), losses)
plt.title("Cross-Entropy Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
