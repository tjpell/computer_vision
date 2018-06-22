
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class LogisticRegression(nn.Module):
	def __init__(self, input_size, num_classes):
		super(LogisticRegression, self).__init__() # why are we supering ourselves?
		self.linear = nn.Linear(input_size, num_classes)


	def forward(self, x):
		out = self.linear(x) # is this what I want? or wrap it in F.log_softmax()? 
		return out

input_dim = 28
input_size = input_dim**2
n_classes = 10
n_epochs = 10
learning_rate = 0.01

# Creating the model object
model = LogisticRegression(input_size, n_classes)

# Initiating loss and optimizer
# do we want SGD or Adam? read up on differences
# want something else for loss? CEL computes softmax internally
loss = nn.CrossEntropyLoss() 

def train_epochs(model, train_dl = train_dl, n_epochs, lr = 0.01, wd = 0.0):
	parameters = filter(lambda p: p.requires_grad, model.parameters()) # is this necessary?
	optimizer = optim.SGD(model.parameters, lr = lr)
	model.train()
	for epoch in torch.range(n_epochs):
		for i, (x, y) in enumerate(train_dl):
			out = model(x)
			loss = F.cross_entropy(out, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if epoch % 2 == 0:
			print('Epoch ' + epoch + ': ' loss.item())
	test_loss(model, test_dl)


def test_loss(model, test_dl = test_dl):
	model.eval()
	total = 0
	correct = 0
	preds_list = list()

	for i, (x, y) in enumerate(test_dl):
		batch = y.shape[0]
		out = model(x)
		loss = F.cross_entropy(out, y)
		preds = torch.argmax(out.data)

		correct += preds.eq(y.data).sum().item()
		total += batch
		preds_list.append(pred)
    print("val loss and accuracy", sum_loss/total, correct/total)
    return preds_list



