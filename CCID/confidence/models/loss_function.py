from torch.nn.modules.loss import _Loss


class SSE(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, under_penalty, over_penalty):
        super(SSE, self).__init__(under_penalty, over_penalty)
        self.under_penalty = under_penalty
        self.over_penalty = over_penalty

    def forward(self, input, target):
        res = ((input - target) ** 2)
        # Scale the loss differently if it's above or below the target
        res[input < target] = res[input < target].mul(self.under_penalty)
        res[input > target] = res[input > target].mul(self.over_penalty)
        return res.sum() / 2
