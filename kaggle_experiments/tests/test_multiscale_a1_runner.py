from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multiscale_a1_runner as multi


class MultiscaleA1Test(unittest.TestCase):
    def test_zero_gate_reproduces_fixed_and_trains(self):
        x = torch.randn(13, 1024)
        torch.manual_seed(17)
        fixed = multi.MultiA1(False)
        torch.manual_seed(17)
        adaptive = multi.MultiA1(True)
        f = fixed(x)
        a = adaptive(x)
        torch.testing.assert_close(f[0], a[0], rtol=0, atol=0)
        torch.testing.assert_close(f[1], a[1], rtol=0, atol=0)
        self.assertEqual(f[3]["indices"].tolist(), a[3]["indices"].tolist())
        torch.testing.assert_close(a[4], torch.full((13, 2), .5))
        loss, _, _, _ = multi.step(adaptive, x, 1, torch.nn.BCEWithLogitsLoss())
        loss.backward()
        self.assertGreater(adaptive.gate.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
