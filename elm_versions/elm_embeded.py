import numpy as np
from scipy.special import softmax
import torch

class ELMClassifier:
    """Probabilistic Output Extreme Learning Machine"""
    def __init__(self, input_size, output_size, model_path):
        # Load the model configuration and state_dict
        model_config = torch.load(model_path, map_location=torch.device('cpu'))
        self.input_size = model_config['input_size']
        self.hidden_layer_size = model_config['hidden_size']
        self.output_size = model_config['output_size']
        
        # Ensure the input and output sizes match
        assert input_size == self.input_size, "Mismatch in input size"
        assert output_size == self.output_size, "Mismatch in output size"
        
        # Extract and set weights and biases for the hidden layer
        hidden_weights = model_config['state_dict']['hidden.weight'].numpy()
        hidden_biases = model_config['state_dict']['hidden.bias'].numpy()
        self.hidden_neurons = [(hidden_weights[i], hidden_biases[i]) for i in range(self.hidden_layer_size)]
        
        
        # Extract and set weights and biases for the output layer
        output_weights = model_config['state_dict']['output.weight'].numpy()
        output_biases = model_config['state_dict']['output.bias'].numpy()
        self.output_weights = output_weights
        self.output_biases = output_biases

    def fit(self, x, y, c=1):
        y[y<0.5] = 0.0001
        y[y>0.5] = 0.9999
        #assert len(x.shape) == 2 and len(y.shape) ==2, 'wrong shape inputs for fit'
        x_features, y_features = x.shape[1], y.shape[1]
        self.H = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
        hth = np.dot(np.transpose(self.H), self.H)
        inv_hth_plus_ic = np.linalg.pinv( hth + np.eye(hth.shape[0]) / c )
        ht_logs = np.dot(np.transpose(self.H), np.log(1 - y) - np.log(y))
        self.beta = -1 * np.dot( inv_hth_plus_ic, ht_logs)

    def predict(self, x):
        self.H = np.asarray([self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
        ret = 1.0 / ( 1 + np.exp(-1* np.dot(self.H, self.beta)))
        sums =  np.sum(ret, axis=1)
        ret1 = ret / sums.reshape(-1,1)
        ret2 = softmax(ret, axis=-1)
        retfinal = np.ones(ret.shape)
        retfinal[sums >=1, :] = ret1[sums>=1, :]
        retfinal[sums < 1, :] = ret2[sums<1, :]
        return np.argmax(retfinal,axis=-1)

    def predict_proba(self, x):
        self.H = np.asarray([self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
        ret = 1.0 / ( 1 + np.exp(-1* np.dot(self.H, self.beta)))
        sums =  np.sum(ret, axis=1)
        ret1 = ret / sums.reshape(-1,1)
        ret2 = softmax(ret, axis=-1)
        retfinal = np.ones(ret.shape)
        retfinal[sums >=1, :] = ret1[sums>=1, :]
        retfinal[sums < 1, :] = ret2[sums<1, :]
        return retfinal

    def _activate(self, a, x, b):
        return 1.0 / (1 + np.exp(-1 * np.dot(a, x.T) + b))

    # def predict(self, x):
    #     self.H = np.asarray([self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
    #     output_activations = np.dot(self.H, self.output_weights.T) + self.output_biases
    #     predictions = softmax(output_activations, axis=1)
    #     return np.argmax(predictions, axis=1)

    # def predict_proba(self, x):
    #     self.H = np.asarray([self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
    #     output_activations = np.dot(self.H, self.output_weights.T) + self.output_biases
    #     probabilities = softmax(output_activations, axis=1)
    #     return probabilities

    def fit_with_mask(self, x, y, prune_rate, c=1):
        y[y < 0.5] = 0.0001
        y[y > 0.5] = 0.9999
        # assert len(x.shape) == 2 and len(y.shape) ==2, 'wrong shape inputs for fit'
        x_features, y_features = x.shape[1], y.shape[1]
        self.H = np.asarray([self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
        self.calculate_mask(self.H, prune_rate)
        hth = np.dot(np.transpose(self.H), self.H)
        inv_hth_plus_ic = np.linalg.pinv(hth + np.eye(hth.shape[0]) / c)
        ht_logs = np.dot(np.transpose(self.H), np.log(1 - y) - np.log(y))
        self.beta = -1 * np.dot(inv_hth_plus_ic, ht_logs)


    def predict_with_mask(self, x):
        h = np.asarray([self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
        h = h * self.prune_mask
        ret = 1.0 / (1 + np.exp(-1 * np.dot(h, self.beta)))
        sums = np.sum(ret, axis=1)
        ret1 = ret / sums.reshape(-1, 1)
        ret2 = softmax(ret, axis=-1)
        retfinal = np.ones(ret.shape)
        retfinal[sums >= 1, :] = ret1[sums >= 1, :]
        retfinal[sums < 1, :] = ret2[sums < 1, :]
        return np.argmax(retfinal, axis=-1)



    def calculate_mask(self, h, prune_rate):
        mean = np.mean(h, axis=0)
        self.prune_mask = np.ones_like(mean)
        number_to_prune = int(prune_rate * len(mean))
        mask_indices = np.argpartition(mean, number_to_prune)[:number_to_prune]
        self.prune_mask[mask_indices] = 0
