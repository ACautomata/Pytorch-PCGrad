import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_pcgrad import PCGrad


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


class PCGradTest(unittest.TestCase):
    def test_backward_sets_gradients_for_shared_network(self):
        torch.manual_seed(4)
        x, y = torch.randn(2, 3), torch.randn(2, 4)
        net = TestNet()
        y_pred = net(x)
        optimizer = PCGrad(optim.Adam(net.parameters()))
        losses = [nn.L1Loss()(y_pred, y), nn.MSELoss()(y_pred, y)]

        optimizer.zero_grad()
        optimizer(losses).backward()

        self.assertTrue(all(p.grad is not None for p in net.parameters()))

    def test_backward_handles_task_specific_heads(self):
        torch.manual_seed(4)
        x, y = torch.randn(2, 3), torch.randn(2, 4)
        net = MultiHeadTestNet()
        y_pred_1, y_pred_2 = net(x)
        optimizer = PCGrad(optim.Adam(net.parameters()))
        losses = [nn.MSELoss()(y_pred_1, y), nn.MSELoss()(y_pred_2, y)]

        optimizer.zero_grad()
        optimizer(losses).backward()

        self.assertTrue(all(p.grad is not None for p in net.parameters()))

    def test_objectives_backward_calls_backward(self):
        torch.manual_seed(4)
        x, y = torch.randn(2, 3), torch.randn(2, 4)
        net = TestNet()
        y_pred = net(x)
        optimizer = PCGrad(optim.Adam(net.parameters()))
        losses = optimizer.objectives([nn.L1Loss()(y_pred, y), nn.MSELoss()(y_pred, y)])

        optimizer.zero_grad()
        losses.backward()

        self.assertTrue(all(p.grad is not None for p in net.parameters()))

    def test_call_returns_objectives_wrapper(self):
        torch.manual_seed(4)
        x, y = torch.randn(2, 3), torch.randn(2, 4)
        net = TestNet()
        y_pred = net(x)
        optimizer = PCGrad(optim.Adam(net.parameters()))
        losses = optimizer([nn.L1Loss()(y_pred, y), nn.MSELoss()(y_pred, y)])

        optimizer.zero_grad()
        losses.backward()

        self.assertTrue(all(p.grad is not None for p in net.parameters()))

    def test_mean_reduction_averages_shared_non_conflicting_gradients(self):
        param = nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = PCGrad(optim.SGD([param], lr=0.1), reduction="mean")
        losses = [param.sum(), (2 * param).sum()]

        optimizer.zero_grad()
        optimizer(losses).backward()

        torch.testing.assert_close(param.grad, torch.tensor([1.5, 1.5]))

    def test_sum_reduction_sums_shared_non_conflicting_gradients(self):
        param = nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = PCGrad(optim.SGD([param], lr=0.1), reduction="sum")
        losses = [param.sum(), (2 * param).sum()]

        optimizer.zero_grad()
        optimizer(losses).backward()

        torch.testing.assert_close(param.grad, torch.tensor([3.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
