"""Focused regression checks for information boundaries and official layer parity."""
import unittest

import torch

from .models import RadioUNet, conditions
from .losses import RMEObjectives, SparsePropagationTemplate


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20260924)

    def test_conditions_do_not_read_target_or_tx(self):
        shape = (2, 1, 1, 128, 128)
        sparse = {key: torch.rand(shape) for key in ('building', 'vehicle', 'observed_rss', 'sampling_mask', 'target', 'tx')}
        expected = conditions(sparse)
        sparse['target'].fill_(float('nan'))
        sparse['tx'].fill_(float('nan'))
        self.assertTrue(torch.equal(expected, conditions(sparse)))
        self.assertEqual(expected[:, 1].abs().sum(), 0)

    def test_official_radiounet_forward_and_stage_freezing(self):
        net = RadioUNet().eval()
        x = torch.rand(1, 5, 128, 128)
        with torch.no_grad():
            expected = net.net(x)[0]
            torch.testing.assert_close(net(x), expected, rtol=0, atol=0)
            net.set_phase('second')
            net.net.phase = 'secondU'
            expected = net.net(x)[1]
            torch.testing.assert_close(net(x), expected, rtol=0, atol=0)
        for name, parameter in net.net.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith('W'))

    def test_sparse_template_cannot_use_unobserved_values(self):
        fitter = SparsePropagationTemplate()
        mask = (torch.rand(2, 1, 128, 128) < .03).float()
        observed = torch.rand_like(mask) * mask
        expected = fitter(observed, mask)
        actual = fitter(observed + (1 - mask) * 1000, mask)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(actual).all())

    def test_local_losses_reach_generator_and_do_not_select_missing_samples(self):
        objective = RMEObjectives()
        x = torch.zeros(2, 5, 128, 128)
        x[:, 4, 20, 30] = 1
        x[:, 3, 20, 30] = .7
        pred = torch.full((2, 1, 128, 128), .2, requires_grad=True)
        geometry = objective.geometry_loss(pred, x[:, 3:4], x[:, 4:5])
        self.assertAlmostEqual(geometry.item(), .25, places=6)
        geometry.backward()
        self.assertEqual(torch.count_nonzero(pred.grad).item(), 2)
        pred.grad = None
        loss, logs = objective(pred, torch.rand_like(pred), x, 'second', pred.mean() * 0)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().sum(), 0)
        self.assertGreater(logs['high_frequency'], 0)


if __name__ == '__main__':
    unittest.main()
