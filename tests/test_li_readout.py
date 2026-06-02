"""TDD tests for LINode and ActionHead with readout='li'."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import node
import networks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LIF_P = {"threshold": 0.5, "tau": 2.0}
NORM_P = {"threshold": 0.5, "v_reset": 0.0}
INP_DIM = 16
NUM_ACTIONS = 4
UNITS = 32
T = 3
BATCH = 2


def make_action_head(readout="linear", dist="trunc_normal"):
    return networks.ActionHead(
        inp_dim=INP_DIM,
        size=NUM_ACTIONS,
        layers=2,
        units=UNITS,
        act="LIFNode",
        act_p=LIF_P,
        norm="PopNorm",
        norm_p=NORM_P,
        dist=dist,
        spike_times=T,
        readout=readout,
        readout_tau=2.0,
    )


def make_features(T=T, batch=BATCH, feat_dim=INP_DIM):
    """Shape (T, batch, feat_dim) — matches how ActionHead.forward indexes features."""
    return torch.randn(T, batch, feat_dim)


# ---------------------------------------------------------------------------
# Block A: LINode unit tests
# ---------------------------------------------------------------------------

class TestLINode:
    def test_single_step_value(self):
        """First forward: mem = 0 + (input - 0) / tau = input / tau."""
        li = node.LINode(tau=2.0)
        x = torch.tensor([[1.0, 2.0, 4.0]])
        out = li(x)
        expected = x / 2.0
        assert torch.allclose(out, expected), f"Expected {expected}, got {out}"

    def test_multi_step_accumulation(self):
        """EMA formula: mem_t = mem_{t-1} + (I - mem_{t-1}) / tau."""
        li = node.LINode(tau=2.0)
        x = torch.tensor([[1.0]])
        # step 1: mem = 0 + (1-0)/2 = 0.5
        li(x)
        # step 2: mem = 0.5 + (1-0.5)/2 = 0.75
        li(x)
        # step 3: mem = 0.75 + (1-0.75)/2 = 0.875
        out = li(x)
        assert torch.allclose(out, torch.tensor([[0.875]])), f"Got {out}"

    def test_reset_clears_membrane(self):
        """n_reset() must set self.mem back to v_reset."""
        li = node.LINode(tau=2.0, v_reset=0.0)
        x = torch.tensor([[1.0, -1.0]])
        li(x)
        assert not isinstance(li.mem, float) or li.mem != 0.0  # membrane changed
        li.n_reset()
        assert li.mem == 0.0, f"After reset mem should be 0.0, got {li.mem}"

    def test_output_equals_mem(self):
        """Return value of forward() must equal self.mem."""
        li = node.LINode(tau=2.0)
        x = torch.randn(2, 8)
        out = li(x)
        assert torch.allclose(out, li.mem), "forward() must return self.mem"

    def test_grad_flows_through_integration(self):
        """Gradient must flow from LINode output back to inputs."""
        li = node.LINode(tau=2.0)
        x = torch.randn(1, 4, requires_grad=True)
        out = li(x)
        out.sum().backward()
        assert x.grad is not None, "x.grad should not be None"
        assert not torch.all(x.grad == 0), "gradient must be non-zero"

    def test_no_spiking_output(self):
        """LINode output must NOT be binary — it is a continuous membrane potential."""
        li = node.LINode(tau=2.0)
        x = torch.randn(8, 32) * 2  # large inputs to trigger hypothetical spikes
        out = li(x)
        # output must have fractional values — not all 0 or 1
        is_binary = torch.all((out == 0) | (out == 1))
        assert not is_binary, "LINode must output continuous values, not spikes"


# ---------------------------------------------------------------------------
# Block B: ActionHead with readout='li'
# ---------------------------------------------------------------------------

class TestActionHeadLI:
    def test_output_shape(self):
        """dist.sample() must have shape (batch, num_actions)."""
        head = make_action_head(readout="li")
        dist = head(make_features())
        sample = dist.sample()
        assert sample.shape == (BATCH, NUM_ACTIONS), f"Got shape {sample.shape}"

    def test_no_nan_in_output(self):
        """Forward must produce finite values (no NaN/Inf)."""
        head = make_action_head(readout="li")
        dist = head(make_features())
        mean = dist.mean
        assert torch.isfinite(mean).all(), f"Non-finite values in mean: {mean}"

    def test_mean_in_valid_range_trunc_normal(self):
        """For trunc_normal, internal mean (before sampling) must be in [-1, 1]."""
        head = make_action_head(readout="li", dist="trunc_normal")
        dist = head(make_features())
        mean = dist.mean
        assert (mean >= -1).all() and (mean <= 1).all(), (
            f"mean out of [-1,1]: min={mean.min():.3f}, max={mean.max():.3f}"
        )

    def test_reset_determinism(self):
        """Two forward calls with the same input must give the same mean (reset at start)."""
        head = make_action_head(readout="li")
        head.eval()
        features = make_features()
        with torch.no_grad():
            mean1 = head(features).mean
            mean2 = head(features).mean
        assert torch.allclose(mean1, mean2), (
            f"Output not deterministic after reset: max diff {(mean1-mean2).abs().max():.6f}"
        )

    def test_no_dist_layer(self):
        """readout='li' must not create _dist_layer; must expose _readout_linear."""
        head = make_action_head(readout="li")
        assert not hasattr(head, "_dist_layer"), "readout='li' must not have _dist_layer"
        assert hasattr(head, "_readout_linear"), "readout='li' must have _readout_linear"

    def test_no_norm_in_last_block(self):
        """Last block must be Linear→LINode with no norm in between."""
        import normalization
        head = make_action_head(readout="li")
        layers = list(head._pre_layers)
        li_idx = next(i for i, m in enumerate(layers) if isinstance(m, node.LINode))
        assert li_idx >= 1, "LINode must not be first"
        assert not isinstance(layers[li_idx - 1], normalization.PopNorm), (
            "No norm should appear directly before LINode in the readout block"
        )

    def test_grad_flows_through_readout(self):
        """loss.backward() must give non-zero gradient for _readout_linear weights."""
        head = make_action_head(readout="li")
        features = make_features()
        dist = head(features)
        sample = dist.sample()
        loss = -dist.log_prob(sample.detach()).mean()
        loss.backward()
        w_grad = head._readout_linear.weight.grad
        assert w_grad is not None, "_readout_linear.weight.grad is None"
        assert not torch.all(w_grad == 0), "_readout_linear.weight.grad is all zeros"

    def test_std_above_min_std(self):
        """Standard deviation must be above min_std everywhere."""
        min_std = 0.1
        head = make_action_head(readout="li", dist="trunc_normal")
        # Access std via the underlying distribution
        features = make_features()
        dist_obj = head(features)
        # SafeTruncatedNormal is the base; ContDist wraps Independent which wraps it
        # dist_obj._dist is Independent, dist_obj._dist.base_dist is SafeTruncatedNormal
        std = dist_obj._dist.base_dist.scale
        assert (std >= min_std).all(), f"std below min_std: min={std.min():.4f}"


class TestActionHeadLinearUnchanged:
    """readout='linear' (default) must still work correctly after the change."""

    def test_output_shape(self):
        head = make_action_head(readout="linear")
        dist = head(make_features())
        assert dist.sample().shape == (BATCH, NUM_ACTIONS)

    def test_no_nan(self):
        head = make_action_head(readout="linear")
        dist = head(make_features())
        assert torch.isfinite(dist.mean).all()

    def test_reset_determinism(self):
        head = make_action_head(readout="linear")
        head.eval()
        features = make_features()
        with torch.no_grad():
            m1 = head(features).mean
            m2 = head(features).mean
        assert torch.allclose(m1, m2)
