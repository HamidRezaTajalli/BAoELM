import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
from datetime import date
class ELM_GD_Classifier(nn.Module):
    """Probabilistic Output Extreme Learning Machine trained with Gradient Descent"""
    def __init__(self, input_size, hidden_size, output_size):
        super(ELM_GD_Classifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

def fit(model, train_loader, learning_rate=0.001, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print('Epoch: [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
    model_config = {
    'input_size': model.input_size,
    'hidden_size': model.hidden_size,
    'output_size': model.output_size,
    'state_dict': model.state_dict()
}
    torch.save(model_config, f'elm_model_with_config_{date.today().strftime("%Y-%m-%d")}.pth')
    return model
    

# def predict(x):
#     h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
#     ret = 1.0 / ( 1 + np.exp(-1* np.dot(h, self.beta)))
#     sums =  np.sum(ret, axis=1)
#     ret1 = ret / sums.reshape(-1,1)
#     ret2 = softmax(ret, axis=-1)
#     retfinal = np.ones(ret.shape)
#     retfinal[sums >=1, :] = ret1[sums>=1, :]
#     retfinal[sums < 1, :] = ret2[sums<1, :]
#     return np.argmax(retfinal,axis=-1)

# def predict_proba(x):
#     h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
#     ret = 1.0 / ( 1 + np.exp(-1* np.dot(h, self.beta)))
#     sums =  np.sum(ret, axis=1)
#     ret1 = ret / sums.reshape(-1,1)
#     ret2 = softmax(ret, axis=-1)
#     retfinal = np.ones(ret.shape)
#     retfinal[sums >=1, :] = ret1[sums>=1, :]
#     retfinal[sums < 1, :] = ret2[sums<1, :]
#     return retfinal

# def _activate(a, x, b):
#     return 1.0 / (1 + np.exp(-1* np.dot(a, x.T) + b) )