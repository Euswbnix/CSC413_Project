"""PilotNet-style convolutional encoder, shared (weight-tied) across timesteps.

Shapes at the model input width of 200, printed from a live forward hook and asserted
in tests/test_encoder_shapes.py -- these ARE the annotations on model-figure panel 1,
so the figure cannot drift from the code:

    (3, 66, 200) -> (24, 31, 98) -> (36, 14, 47) -> (48, 5, 22)
                 -> (64, 3, 20)  -> (64, 1, 18)  -> flatten 1152 -> FC 32

Parameter count: 131,348 conv + 36,896 FC = 168,244.

NOTE on the flatten width: Bojarski et al. (2016) state 1164 for this layer, which is
inconsistent with their own architecture figure (64 x 1 x 18 = 1152). We report our own
measured shapes throughout. Do not "fix" 1152 to 1164.

NOTE on storage width: frames are stored at 66x240 and a random 200-wide crop is taken
in the DATA pipeline (REVIEW B4), so the model always sees 200. The assert in forward()
is load-bearing: at width 240 the flatten becomes 1472 and the encoder silently grows to
178,484 parameters, at which point the hand-derived column of the parameter table stops
matching the torch column -- and that match is the only thing proving the table is real.
"""

import torch
import torch.nn as nn

MODEL_INPUT_HW = (66, 200)
FLATTEN_WIDTH = 1152
FEATURE_DIM = 32


class PilotNetEncoder(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, dropout=0.0):
        """`dropout` defaults to 0.0, which reproduces every run collected so far.

        It exists because this encoder is 168,244 of the model's 193,141 parameters -- 87% --
        and carries no dropout and no normalisation, while the project's parameter matching
        was spent entirely on the 12.9% that is the recurrent cell. Its training data is one
        continuous drive in which adjacent frames are near-duplicates, so the effective
        sample count is far below the nominal 37,995. Measured consequence: over 30 epochs
        train loss falls 58% while validation MSE nearly doubles (0.285 -> 0.482) and
        validation global MAE rises 9.96 -> 13.41 deg. Early stopping then fires at a median
        epoch of 4 out of 30, and 15% of runs never improve on epoch 0 at all -- which is why
        the collected models read as near-constant predictors.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3, stride=1), nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ELU(),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(FLATTEN_WIDTH, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, frames):
        """(B, T, 3, 66, 200) -> (B, T, feature_dim). Weights are shared across T by
        folding time into the batch dimension -- that fold IS the weight sharing."""
        assert frames.dim() == 5, f"expected (B,T,C,H,W); got {tuple(frames.shape)}"
        B, T = frames.shape[:2]
        assert frames.shape[2:] == (3,) + MODEL_INPUT_HW, (
            f"expected (3,66,200) per frame; got {tuple(frames.shape[2:])}. "
            "Frames are stored at 66x240; the 200-wide crop belongs in the data pipeline."
        )
        x = frames.reshape(B * T, *frames.shape[2:])
        x = self.conv(x).flatten(1)
        assert x.shape[1] == FLATTEN_WIDTH, f"flatten is {x.shape[1]}, expected {FLATTEN_WIDTH}"
        return self.fc(self.drop(x)).reshape(B, T, self.feature_dim)


class TwoFrameEncoder(PilotNetEncoder):
    """Non-recurrent control with exactly two frames of context: frames t-1 and t stacked
    as 6 channels. Only conv1 changes (1,824 -> 3,624; encoder 170,044). It supplies a
    MECHANISM for a null result -- if two frames is all the temporal signal there is, no
    recurrent architecture can beat it by much."""

    def __init__(self, feature_dim=FEATURE_DIM, dropout=0.0):
        super().__init__(feature_dim, dropout=dropout)
        self.conv[0] = nn.Conv2d(6, 24, 5, stride=2)

    def forward(self, frames):
        assert frames.dim() == 5
        prev = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)   # t-1, edge-replicated at t=0
        stacked = torch.cat([prev, frames], dim=2)                 # (B,T,6,66,200)
        B, T = stacked.shape[:2]
        x = stacked.reshape(B * T, *stacked.shape[2:])
        x = self.conv(x).flatten(1)
        assert x.shape[1] == FLATTEN_WIDTH
        return self.fc(self.drop(x)).reshape(B, T, self.feature_dim)
