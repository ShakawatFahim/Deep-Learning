import torch
import torch.nn as nn
X= torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y= torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test=torch.tensor([5], dtype=torch.float32)
n_sample,n_features=X.shape
print(n_sample,n_features)

input_size=n_features
output_size= n_features

model= nn.Linear(input_size,output_size)
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')
n_iters = 10000

loss=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters() ,lr=0.01)

for epoch in range(n_iters):
    y_pred = model(X)
    l = loss(Y,y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch %100 ==0:
        [w,b]=model.parameters()
        print(f'epoch{epoch+1}: w={w[0][0].item():.3f}, loss={l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')


