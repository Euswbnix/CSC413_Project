"""The arm interface and factory.

Written BEFORE the cell, on purpose: a single `rnn` slot behind one `SteeringRegressor`
is what makes the escape hatch one line. Because our CfCCell is bit-identical to
`ncps.torch.CfCCell(I, H, mode="default", backbone_units=0, backbone_layers=0)` with the
same trainable count (24,832 == 24,832), the cell can be swapped for the library's at any
point WITHOUT RE-RUNNING A SINGLE EXPERIMENT -- see tests/test_cfc_oracle.py. Swap only
the CELL and keep our 16-line wrapper: ncps' `timespans` bug lives in its wrapper
(`timespans[:, t].squeeze()`), not its cell.

Every arm shares, byte for byte: the CNN encoder class, the readout structure, the
optimiser, gradient clipping, the data pipeline and the evaluation path. The intended
difference is the recurrent update equation and nothing else.

Recurrent-slot contract -- every module in the `rnn` slot has the SAME signature:
    forward(z: (B,T,I), dt: (B,T) | None, hx: state | None) -> (out: (B,T,W), state)
`hx` support is mandatory because the evaluation protocol is a stateful chunked rollout
over contiguous test segments.
"""

import torch
import torch.nn as nn

from .cfc import CfC
from .encoder import PilotNetEncoder, TwoFrameEncoder, FEATURE_DIM

HIDDEN = 64          # fixed by rule and stated, NOT swept (PROTOCOL M5 locks lr, T, k)


# ---------------------------------------------------------------- parameter arithmetic

def cfc_params(I, H):                 # 4 heads, each Linear(I+H -> H)
    return 4 * H * (I + H + 1)


def lstm_params(I, H):                # torch carries two bias vectors (cuDNN), hence +8H
    return 4 * H * (I + H) + 8 * H


def gru_params(I, H):
    return 3 * H * (I + H) + 6 * H


def mlp_params(I, D):                 # Linear(I->D) + Linear(D->1)
    return I * D + D + D + 1


def matched_width(target, fn, I, lo=1, hi=4096):
    """Width whose parameter count is CLOSEST to `target`, ties going to the LARGER.

    Closest, not largest-under: the same principle as the LSTM's +4H residual. Overshooting
    slightly is in the BASELINE's favour, so the comparison can never be read as "you
    shrank your opponent". At I=32 against the CfC's 24,832: GRU H=76 gives 25,080
    (+1.00%), while H=75 would give 24,525 (-1.24%) -- worse on both counts.
    """
    best, best_err = lo, abs(fn(I, lo) - target)
    for w in range(lo + 1, hi):
        err = abs(fn(I, w) - target)
        if err <= best_err:
            best, best_err = w, err
        elif fn(I, w) > target:
            break
    return best


# ------------------------------------------------------------------- recurrent adapters

class TorchRNNAdapter(nn.Module):
    """Wraps nn.LSTM / nn.GRU into the recurrent-slot contract.

    `batch_first=True` is passed EXPLICITLY: ncps defaults to True and torch defaults to
    False, and a silent batch/time transpose would void the controlled comparison.

    dt is accepted and IGNORED by default. `use_dt=True` appends it as an input feature --
    the honest GRU-D-style baseline, so "has the information" can be separated from "can
    use it". Report both.
    """

    def __init__(self, kind, input_size, hidden_size, use_dt=False):
        super().__init__()
        cls = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.use_dt = use_dt
        self.rnn = cls(input_size + (1 if use_dt else 0), hidden_size, batch_first=True)
        self.output_width = hidden_size

    def forward(self, z, dt=None, hx=None):
        if self.use_dt:
            if dt is None:
                dt = z.new_ones(z.shape[0], z.shape[1])
            z = torch.cat([z, dt.unsqueeze(-1)], dim=-1)
        return self.rnn(z, hx)


class CfCAdapter(nn.Module):
    def __init__(self, input_size, hidden_size, **kw):
        super().__init__()
        self.rnn = CfC(input_size, hidden_size, **kw)
        self.output_width = hidden_size

    def forward(self, z, dt=None, hx=None):
        return self.rnn(z, dt=dt, hx=hx)


class CausalMean(nn.Module):
    """Cumulative mean of the encoder features: output_t = mean(z_1..z_t).

    The control that turns a mechanistic story into a falsifiable prediction. Measured over
    ten seeds, destroying frame ORDER at training time changed nothing (curve-bin MAE 29.54
    shuffled against 30.31 ordered) while the recurrent arms still beat the per-frame CNNs
    (30-31 against 34-36). The only reading that fits both is that the recurrent layer is
    exploiting the MULTIPLICITY of frames, not their sequence -- i.e. averaging.

    So: same information as the recurrent arms (causal, prefix only -- averaging the whole
    window would leak future frames into early predictions), same readout, order-invariant
    by construction, and ZERO parameters. If this matches the CfC, a 24,832-parameter
    recurrent layer is doing what a running mean does for free, and that is the finding.
    """

    def __init__(self, input_size):
        super().__init__()
        self.output_width = input_size

    def forward(self, z, dt=None, hx=None):
        counts = torch.arange(1, z.shape[1] + 1, device=z.device, dtype=z.dtype)
        run = z.cumsum(dim=1) if hx is None else (z.cumsum(dim=1) + hx[0])
        n = counts.view(1, -1, 1) if hx is None else (counts.view(1, -1, 1) + hx[1])
        out = run / n
        # state = (running sum, count) so a chunked stateful rollout is exact
        return out, (run[:, -1:], n[:, -1:])


class NoRecurrence(nn.Module):
    """Identity in the recurrent slot. The readout then acts per frame with no state, so
    the arm is length-invariant BY CONSTRUCTION -- which is why its per-position MAE curve
    must come out flat, and why a non-flat one means the position accounting is broken."""

    def __init__(self, input_size):
        super().__init__()
        self.output_width = input_size

    def forward(self, z, dt=None, hx=None):
        return z, None


# ------------------------------------------------------------------------- the regressor

class SteeringRegressor(nn.Module):
    def __init__(self, encoder, rnn, readout):
        super().__init__()
        self.encoder, self.rnn, self.readout = encoder, rnn, readout

    def forward(self, frames, dt=None, hx=None):
        """(B,T,3,66,200) -> (B,T,1) in STANDARDISED target units.

        Targets are standardised on train-split mean/std and inverted at report time
        (REVIEW B1). This is not only about loss scale: the recurrent state is strictly
        bounded to (-1,1), and fitting a tanh-bounded state against degree-scale labels
        with a +-100 tail reaches only 87.2 deg after 4,000 AdamW steps versus 99.8 deg on
        standardised targets -- a SCALE problem that looks exactly like "the recurrent
        cell cannot handle sharp turns", i.e. a fabricated architectural finding.
        """
        z = self.encoder(frames)
        out, state = self.rnn(z, dt=dt, hx=hx)
        return self.readout(out), state

    def param_breakdown(self):
        def n(m):
            return sum(p.numel() for p in m.parameters())
        return {"encoder": n(self.encoder), "recurrent": n(self.rnn),
                "readout": n(self.readout), "total": n(self)}


# ----------------------------------------------------------------------------- the arms

ARMS = ("cfc", "lstm", "lstm_dt", "gru", "cnn_mlp", "cnn_linear", "cnn_2frame", "cnn_avg")


def build_arm(name, hidden=HIDDEN, feature_dim=FEATURE_DIM, dropout=0.0, **cfc_kw):
    """Every arm's recurrent block is matched to the CfC's parameter count, except the two
    deliberate brackets (cnn_linear from below, and the CfC itself as the reference)."""
    target = cfc_params(feature_dim, hidden)          # 24,832 at I=32, H=64

    if name == "cfc":
        rnn = CfCAdapter(feature_dim, hidden, **cfc_kw)
    elif name in ("lstm", "lstm_dt"):
        # Same hidden width as the CfC. The residual is exactly 4H = 256 params (1.03%),
        # nothing but torch's redundant second bias vector -- and it is IN THE BASELINE'S
        # FAVOUR, so we did not shrink our opponent. One run therefore serves BOTH of
        # M9's roles: the parameter-matched baseline and the unconstrained anti-straw-man.
        rnn = TorchRNNAdapter("lstm", feature_dim, hidden, use_dt=(name == "lstm_dt"))
    elif name == "gru":
        rnn = TorchRNNAdapter("gru", feature_dim, matched_width(target, gru_params, feature_dim))
    elif name == "cnn_avg":
        rnn = CausalMean(feature_dim)
    elif name in ("cnn_mlp", "cnn_linear", "cnn_2frame"):
        rnn = NoRecurrence(feature_dim)
    else:
        raise ValueError(f"unknown arm {name!r}; expected one of {ARMS}")

    encoder = (TwoFrameEncoder(feature_dim, dropout=dropout) if name == "cnn_2frame"
               else PilotNetEncoder(feature_dim, dropout=dropout))

    if name in ("cnn_mlp", "cnn_avg"):
        # Capacity-matched non-recurrent head: brackets the recurrent arms from ABOVE in
        # head capacity, so "the cost of deleting temporal state" cannot be confused with
        # "the cost of deleting head capacity".
        width = matched_width(target, mlp_params, feature_dim)
        readout = nn.Sequential(nn.Linear(feature_dim, width), nn.ELU(), nn.Linear(width, 1))
    else:
        readout = nn.Linear(rnn.output_width, 1)

    return SteeringRegressor(encoder, rnn, readout)
