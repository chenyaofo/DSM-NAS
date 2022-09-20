import functools

import torch
import torch.nn as nn


class BaseController(nn.Module):
    def __init__(self, hidden_size, device):
        super(BaseController, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

    def reset_parameters(self, init_range=0.1):
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    @functools.lru_cache(maxsize=128)
    def _zeros(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), device=self.device, requires_grad=False)

    def _scale_attention(self, logits, temperature, tanh_constant, constant_reduce=None):
        if temperature is not None:
            logits /= temperature
        if tanh_constant is not None:
            if constant_reduce is not None:
                tanh_constant /= constant_reduce
            logits = tanh_constant * torch.tanh(logits)
        return logits

    def _impl(self, probs):
        m = torch.distributions.Categorical(probs=probs)
        action = m.sample().view(-1)
        select_log_p = m.log_prob(action)
        entropy = m.entropy()
        return action, select_log_p, entropy

    def get_self_device(self):
        return list(self.parameters)[0].device
