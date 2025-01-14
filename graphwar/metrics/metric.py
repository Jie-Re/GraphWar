import torch


class Metric(torch.nn.Module):

    def __init__(self, name=None, **kwargs):
        super().__init__()
        self.name = name or self.__class__.__name__
        self.reset_states()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def update_state(self):
        raise NotImplementedError

    def reset_states(self):
        raise NotImplementedError

    def reset_state(self):
        return self.reset_states()

    def result(self):
        raise NotImplementedError

    def extra_repr(self):
        return f"name={self.name}"
