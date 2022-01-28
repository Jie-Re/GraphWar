import torch.nn as nn
from torch.nn import Linear
from graphwar.nn import Sequential, activations, APPNPConv
from graphwar.config import Config

_EDGE_WEIGHT = Config.edge_weight


class APPNP(nn.Module):
    """Approximated personalized propagation
    of neural predictions (APPNP)

    Example
    -------
    # APPNP with one hidden layer
    >>> model = APPNP(100, 10)
    # APPNP with two hidden layers
    >>> model = APPNP(100, 10, hids=[32, 16], acts=['relu', 'elu'])
    # APPNP with two hidden layers, without activation at the first layer
    >>> model = APPNP(100, 10, hids=[32, 16], acts=[None, 'relu'])
    """

    def __init__(self,
                 in_feats: int,
                 out_feats: int,
                 hids: list = [64],
                 acts: list = ['relu'],
                 dropout: float = 0.8,
                 k: int = 10,
                 alpha: float = 0.1,
                 bn: bool = False,
                 bias: bool = True,
                 norm: str = 'both'):
        r"""
        Parameters
        ----------
        in_feats : int, 
            the input dimmensions of model
        out_feats : int, 
            the output dimensions of model
        hids : list, optional
            the number of hidden units of each hidden layer, by default [64]
        acts : list, optional
            the activaction function of each hidden layer, by default ['relu']
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        bias : bool, optional
            whether to use bias in the layers, by default True
        bn: bool, optional
            whether to use `BatchNorm1d` after the convolution layer, by default False
        norm : str, optional
            How to apply the normalizer.  Can be one of the following values:

            * ``both``, where the messages are scaled with :math:`1/c_{ji}`, 
            where :math:`c_{ji}` is the product of the square root of node degrees
            (i.e.,  :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`).

            * ``square``, where the messages are scaled with :math:`1/c_{ji}^2`, where
            :math:`c_{ji}` is defined as above.

            * ``right``, to divide the aggregated messages by each node's in-degrees,
            which is equivalent to averaging the received messages.

            * ``none``, where no normalization is applied.

            * ``left``, to divide the messages sent out from each node by its out-degrees,
            equivalent to random walk normalization.                   
        """

        super().__init__()
        assert len(hids) > 0

        lin = []
        for hid, act in zip(hids, acts):
            lin.append(nn.Dropout(dropout))
            lin.append(Linear(in_feats, hid, bias=bias))
            if bn:
                lin.append(nn.BatchNorm1d(hid))
            in_feats = hid
            lin.append(activations.get(act))

        lin.append(nn.Dropout(dropout))
        lin.append(Linear(in_feats, out_feats, bias=bias))

        self.prop = APPNPConv(k, alpha, norm=norm)
        self.lin = Sequential(*lin)

    def reset_parameters(self):
        self.prop.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, g, feat, edge_weight=None):
        feat = self.lin(feat)
        if edge_weight is None:
            edge_weight = g.edata.get(_EDGE_WEIGHT, edge_weight)
        feat = self.prop(g, feat, edge_weight=edge_weight)
        return feat
