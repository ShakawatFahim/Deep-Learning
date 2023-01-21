import torch
import torch.nn as nn
#import torch.nn.functional as F

#option 1(create nn faction/module)
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNet,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        #nn.LeakyReLU()
        #nn.Tanh()
        #nn.Softmax()
    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.Linear2(out)
        out = self.sigmoid(out)
        return out


#option 2(use activation directly in forward pass)
class NeuralNet1(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = self.Linear(input_size,hidden_size)
        self.linear2 = self.Linear(hidden_size, 1)

    def forward(self,x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        #torch.softmax()/F.softmax()
        #torch.tanh()/F.tanh()
        #F.leaky_relu() :it doesnot has the direct torch function so thats why we need torch.nn.functional
        return out


# output = w*x + b
# output = activation_function(output)


# sofmax
output = torch.softmax(x, dim=0)
print(output)
sm = nn.Softmax(dim=0)
output = sm(x)
print(output)

# sigmoid
output = torch.sigmoid(x)
print(output)
s = nn.Sigmoid()
output = s(x)
print(output)

# tanh
output = torch.tanh(x)
print(output)
t = nn.Tanh()
output = t(x)
print(output)

# relu
output = torch.relu(x)
print(output)
relu = nn.ReLU()
output = relu(x)
print(output)

# leaky relu
output = F.leaky_relu(x)
print(output)
lrelu = nn.LeakyReLU()
output = lrelu(x)
print(output)


# nn.ReLU() creates an nn.Module which you can add e.g. to an nn.Sequential model.
# torch.relu on the other side is just the functional API call to the relu function,
# so that you can add it e.g. in your forward method yourself.

# option 1 (create nn modules)
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


# option 2 (use activation functions directly in forward pass)
class NeuralNet3(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet3, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out