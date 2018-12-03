import torch.nn as nn
import torch


class FCN(nn.Module):

    def __init__(self, encoded_vector_len, in_features, hidden_neurons=[], bias=True, apply_dropout=0.5):
        """
        This is a nn.Module class which defines the layer architecture of the Classifier module.
        This is a Fully Connected Network classifier which has hidden layer and a final classification layer.
        params:
            classes (list): list of class names that the model will be trained on
            in_features (int): size of the flattened feature map from the Feature Extractor that will be connected to this classifier
            hidden_neurons (list): if this is empty then a 1FCN is created consisting of a single classification layer.
                If this list is not empty, then a hidden layers are added to the model based on the number of neurons defined in the list.
                For example:
                    if hidden_neurons = [1024, 512] then two fully connected layers are added which have 1024, 512 output neurons consequently.
            bias (bool): apply bias to the fully connected layers
            apply_dropout (float): if this is None, then no dropout is applied. Otherwise dropout with probability=apply_dropout is applied to the network after each hidden layer.
        returns:
            nn.Module
        """
        super(FCN, self).__init__()

        self.cuda_enabled = torch.cuda.is_available()

        self.vector_length = encoded_vector_len
        self.apply_dropout = apply_dropout

        self.bias = bias

        hidden_layers = []
        num_features = [in_features] + hidden_neurons

        if hidden_neurons:
            for i in range(len(hidden_neurons)):
                hidden_layers += self.create_hidden_layer(
                    in_features=num_features[i], out_features=num_features[i+1])

        hidden_layers += [nn.Linear(num_features[-1],
                                    self.vector_length, bias=True)]

        self.model = nn.Sequential(*hidden_layers)

        # make the classifier run on gpu
        if self.cuda_enabled:
            self.cuda()

    def create_hidden_layer(self, in_features, out_features):
        layers = [nn.Linear(in_features, out_features,
                            bias=self.bias), nn.ReLU()]

        if self.apply_dropout:
            layers += [nn.Dropout(p=self.apply_dropout)]

        return layers

    def forward(self, x):
        return self.model(x)
