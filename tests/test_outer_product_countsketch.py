import sys
import unittest
from pathlib import Path
from typing import Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from selection.influence_scoring import OuterProductCountSketch  # noqa: E402


def manual_outer_product_sketch(
    sketcher: OuterProductCountSketch,
    activations: torch.Tensor,
    backprops: torch.Tensor,
    preconditioner: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = torch.zeros(activations.size(0), sketcher.m, dtype=torch.float32)
    for batch_idx in range(activations.size(0)):
        for out_idx in range(sketcher.d_out):
            for in_idx in range(sketcher.d_in):
                bucket = int(
                    (sketcher.h_out[out_idx] + sketcher.h_in[in_idx]).remainder(
                        sketcher.m
                    )
                )
                value = (
                    backprops[batch_idx, out_idx]
                    * activations[batch_idx, in_idx]
                    * sketcher.s_out[out_idx]
                    * sketcher.s_in[in_idx]
                )
                if preconditioner is not None:
                    value = value * preconditioner[out_idx, in_idx]
                out[batch_idx, bucket] += value
    return out


class OuterProductCountSketchTest(unittest.TestCase):
    def test_tensorsketch_matches_direct_unweighted_outer_product(self):
        torch.manual_seed(0)
        sketcher = OuterProductCountSketch(
            d_in=7,
            d_out=5,
            m=17,
            seed=123,
            row_block=2,
        )
        activations = torch.randn(4, 7)
        backprops = torch.randn(4, 5)

        fast = sketcher.sketch_outer(activations, backprops)
        direct = sketcher._sketch_outer_direct(
            activations.to(torch.float32),
            backprops.to(torch.float32),
        )
        manual = manual_outer_product_sketch(sketcher, activations, backprops)

        torch.testing.assert_close(fast, direct, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(fast, manual, rtol=1e-5, atol=1e-5)

    def test_weighted_path_preserves_coordinatewise_preconditioner(self):
        torch.manual_seed(1)
        sketcher = OuterProductCountSketch(
            d_in=6,
            d_out=4,
            m=13,
            seed=456,
            row_block=3,
        )
        activations = torch.randn(3, 6)
        backprops = torch.randn(3, 4)
        preconditioner = torch.randn(4, 6)

        actual = sketcher.sketch_outer(activations, backprops, preconditioner)
        expected = manual_outer_product_sketch(
            sketcher,
            activations,
            backprops,
            preconditioner,
        )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_tokenwise_sum_matches_sequence_gradient_sketch(self):
        torch.manual_seed(2)
        sketcher = OuterProductCountSketch(
            d_in=5,
            d_out=3,
            m=19,
            seed=789,
            row_block=2,
        )
        batch_size = 2
        seq_len = 4
        activations = torch.randn(batch_size, seq_len, 5)
        backprops = torch.randn(batch_size, seq_len, 3)

        token_sketches = sketcher.sketch_outer(
            activations.reshape(batch_size * seq_len, 5),
            backprops.reshape(batch_size * seq_len, 3),
        ).reshape(batch_size, seq_len, -1)
        actual = token_sketches.sum(dim=1)

        expected = torch.zeros(batch_size, sketcher.m, dtype=torch.float32)
        for token_idx in range(seq_len):
            expected += manual_outer_product_sketch(
                sketcher,
                activations[:, token_idx, :],
                backprops[:, token_idx, :],
            )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
