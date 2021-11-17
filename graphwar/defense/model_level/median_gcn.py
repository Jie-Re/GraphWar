import torch.nn as nn
from graphwar.nn import Sequential, activations
from graphwar.nn import MedianConv


class MedianGCN(nn.Module):
    """Graph Convolution Network (GCN) with `median` aggregation function.

    Example
    -------
    # MedianGCN with one hidden layer
    >>> model = MedianGCN(100, 10)
    # MedianGCN with two hidden layers
    >>> model = MedianGCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])
    # MedianGCN with two hidden layers, without activation at the first layer
    >>> model = MedianGCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    Note
    ----
    please make sure `hids` and `acts` are both `list` or `tuple` and
    `len(hids)==len(acts)`.

    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hids: list = [16],
                 acts: list = ['relu'],
                 dropout: float = 0.5,
                 bias: bool = True,
                 norm: str = 'none'):
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
        bias : bool, optional
            whether to use bias in the layers, by default True
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

        conv = []
        assert len(hids) == len(acts)
        for hid, act in zip(hids, acts):
            conv.append(MedianConv(in_features,
                                   hid,
                                   bias=bias, norm=norm,
                                   activation=activations.get(act)))
            conv.append(nn.Dropout(dropout))
            in_features = hid
        conv.append(MedianConv(in_features, out_features, bias=bias, norm=norm))
        self.conv = Sequential(*conv, loc=1)  # `loc=1` specifies the location of features.

    def reset_parameters(self):
        for conv in self.conv:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()

    def forward(self, g, feat, edge_weight=None):

        for conv in self.conv:
            if isinstance(conv, MedianConv):
                feat = conv(g, feat, edge_weight=edge_weight)
            else:
                feat = conv(feat)
        return feat
