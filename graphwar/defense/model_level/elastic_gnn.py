import torch
import torch.nn as nn
from torch.nn import Linear, Sequential
from graphwar.nn import activations
from graphwar.functional import EMP
from torch_sparse import SparseTensor


class ElasticGNN(nn.Module):
    """Elastic Graph Neural Networks.

    Example
    -------
    # ElasticGNN with one hidden layer
    >>> model = ElasticGNN(100, 10)
    # ElasticGNN with two hidden layers
    >>> model = ElasticGNN(100, 10, hids=[32, 16], acts=['relu', 'elu'])
    # ElasticGNN with two hidden layers, without activation at the first layer
    >>> model = ElasticGNN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    Note
    ----
    please make sure `hids` and `acts` are both `list` or `tuple` and
    `len(hids)==len(acts)`.

    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hids: list = [64],
                 acts: list = ['relu'],
                 dropout: float = 0.8,
                 ):
        r"""
        Parameters
        ----------
        in_features : int, 
            the input dimmensions of model
        out_features : int, 
            the output dimensions of model
        hids : list, optional
            the number of hidden units of each hidden layer, by default [16]
        acts : list, optional
            the activaction function of each hidden layer, by default ['relu']
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        """

        super().__init__()
        assert len(hids) == len(acts) and len(hids) > 0

        conv = []
        for hid, act in zip(hids, acts):
            conv.append(nn.Dropout(dropout))
            conv.append(Linear(in_features, hid))
            in_features =hid
            conv.append(activations.get(act))

        conv.append(nn.Dropout(dropout))
        conv.append(Linear(in_features, out_features))

        self.prop = EMP(K=1,
                        lambda1=3,
                        lambda2=3,
                        L21=True,
                        cached=True)

        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, g, feat):
        feat = self.conv(feat)

        adj = g.adj(scipy_fmt='coo')
        s_row = torch.tensor(adj.row, dtype=torch.long, device=g.device)
        s_col = torch.tensor(adj.col, dtype=torch.long, device=g.device)
        sparse_adj = SparseTensor(row=s_col, col=s_row)

        feat = self.prop(feat, sparse_adj)
        return nn.functional.log_softmax(feat, dim=1)

