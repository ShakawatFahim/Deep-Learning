import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt

X_numpy, y_numpy=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features=X.shape
#model
input_size=n_features
output_size=1
model=nn.Linear(input_size,output_size)
#loss and optimizer
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
#training loop
num_epochs=100
for epoch in range(num_epochs):
    #forward pass
    y_predict = model(X)
    loss = criterion(y_predict,y)
    #backward pass
    loss.backward()
    #update
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1) % 10 ==0:
        print(f"epoch:{epoch+1},loss={loss.item():.4f}")
#plot
predicted=model(X).detach().numpy()
plt.plot(X_numpy, y_numpy,'bo')
plt.plot(X_numpy, predicted,'black')
plt.show()

