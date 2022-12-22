import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
#import matplotlib.pyplot as plt

#data prepare

bc= datasets.load_breast_cancer()
X,y = bc.data, bc.target
n_sample,n_features = X.shape
#print(n_sample,n_features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

#model
class Logistic_Regration(nn.Module):

    def __init__(self,n_input_features):
        super(Logistic_Regration, self).__init__()
        self.linear = nn.Linear(n_input_features,1)

    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = Logistic_Regration(n_features)

#loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#training loops

iteration = 100

for epoch in range(iteration):
    #forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted,y_train)
    #backward pass
    loss.backward()
    #updates
    optimizer.step()
    #Zero grads
    optimizer.zero_grad()

    if (epoch+1)%10==0:
        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')

with torch.no_grad():
    y_prediction = model(X_test)
    y_prediction_cls = y_prediction.round()
    acc = y_prediction_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc.item():.4f}')
#plot
#predicted=model(X_train).detach().numpy()
#plt.plot(X_train, y_train,'bo')
#plt.plot(X_train, predicted,'black')
#plt.show()