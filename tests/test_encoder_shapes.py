"""The asserted shapes ARE model-figure panel 1's annotations, so the figure cannot drift
from the code. Generate the figure from this test's output, not from memory."""

import pytest
import torch

from models.encoder import PilotNetEncoder, FLATTEN_WIDTH, MODEL_INPUT_HW

EXPECTED = [(24, 31, 98), (36, 14, 47), (48, 5, 22), (64, 3, 20), (64, 1, 18)]


def conv_shapes(enc, width=200):
    shapes, x = [], torch.zeros(1, 3, 66, width)
    for layer in enc.conv:
        x = layer(x)
        if isinstance(layer, torch.nn.Conv2d):
            shapes.append(tuple(x.shape[1:]))
    return shapes, x.flatten(1).shape[1]


def test_conv_shape_chain_and_flatten():
    shapes, flat = conv_shapes(PilotNetEncoder())
    assert shapes == EXPECTED
    # Bojarski et al. state 1164 for this layer, which contradicts their own figure
    # (64 x 1 x 18 = 1152). We report our own measured shapes. Do not "fix" this.
    assert flat == FLATTEN_WIDTH == 1152


def test_forward_rejects_the_storage_width():
    """Frames are STORED at 66x240 and cropped to 200 in the data pipeline (REVIEW B4).
    At width 240 the flatten becomes 1472 and the encoder silently grows to 178,484
    parameters -- at which point the hand-derived column of the parameter table stops
    matching the torch column, and that match is the only proof the table is real."""
    _, flat240 = conv_shapes(PilotNetEncoder(), width=240)
    assert flat240 == 1472
    with pytest.raises(AssertionError):
        PilotNetEncoder()(torch.zeros(1, 2, 3, 66, 240))


def test_weights_are_shared_across_timesteps():
    """The fold of time into the batch dimension IS the weight sharing: the same frame at
    two different positions must produce identical features."""
    enc = PilotNetEncoder().double().eval()
    g = torch.Generator().manual_seed(2)
    frame = torch.randn(1, 1, 3, *MODEL_INPUT_HW, generator=g, dtype=torch.float64)
    out = enc(frame.repeat(1, 4, 1, 1, 1))
    for t in range(1, 4):
        # NOT atol=0, and the tolerance is derived rather than tuned. Convolution reduces,
        # and a batched GEMM may accumulate identical rows in a different order, so the same
        # frame at two positions can differ in the last bit (measured 4e-17 in float64 on the
        # 32-thread training box; bit-identical on a 10-thread laptop, which is environment
        # luck, not a guarantee). What this test distinguishes is weight SHARING: per-timestep
        # weights would differ by O(1). 1e-14 leaves fourteen orders of margin.
        torch.testing.assert_close(out[:, t], out[:, 0], rtol=0, atol=1e-14)
