
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
		out = self.linear(x)
		return out


# Creating the model object
model = LogisticRegression(input_size, num_classes)

# Initiating loss and optimizer
# do we want SGD or Adam? read up on differences
# want something else for loss? CEL computes softmax internally
loss = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr = learning_rate)