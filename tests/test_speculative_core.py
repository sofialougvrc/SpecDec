from __future__ import annotations

import unittest
from random import Random

from specdec.adaptive import AdaptiveSpeculationController
from specdec.core import SpeculativeDecoder, autoregressive_generate
from specdec.distributions import acceptance_probability, positive_difference_distribution
from specdec.models import BigramModel, default_toy_pair


class SpeculativeCoreTests(unittest.TestCase):
    def test_step_samples_first_token_from_target_distribution_analytically(self) -> None:
        target = [0.10, 0.60, 0.30]
        draft = [0.30, 0.40, 0.30]
        emitted = [0.0, 0.0, 0.0]

        for token, q_token in enumerate(draft):
            alpha = acceptance_probability(target[token], q_token)
            emitted[token] += q_token * alpha
            residual = positive_difference_distribution(target, draft)
            for out_token, residual_prob in enumerate(residual):
                emitted[out_token] += q_token * (1.0 - alpha) * residual_prob

        for actual, expected in zip(emitted, target):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_generate_respects_requested_length(self) -> None:
        target, draft = default_toy_pair()
        decoder = SpeculativeDecoder(target=target, draft=draft, rng=Random(4))
        output, stats = decoder.generate([0], max_new_tokens=17, depth=5)
        self.assertEqual(len(output), 18)
        self.assertEqual(stats.emitted_tokens, 17)
        self.assertGreater(stats.target_calls, 0)
        self.assertGreater(stats.draft_calls, 0)

    def test_zero_max_new_tokens_returns_prefix(self) -> None:
        target, draft = default_toy_pair()
        decoder = SpeculativeDecoder(target=target, draft=draft, rng=Random(4))
        output, stats = decoder.generate([0, 1], max_new_tokens=0, depth=5)
        self.assertEqual(output, [0, 1])
        self.assertEqual(stats.emitted_tokens, 0)

    def test_controller_moves_depth_with_acceptance_rate(self) -> None:
        controller = AdaptiveSpeculationController(initial_depth=3, min_depth=1, max_depth=5)
        controller.observe(3, 3)
        self.assertEqual(controller.current_depth, 4)
        for _ in range(4):
            controller.observe(0, 4)
        self.assertLess(controller.current_depth, 4)

    def test_autoregressive_smoke(self) -> None:
        target = BigramModel(((0.0, 1.0), (1.0, 0.0)))
        output, stats = autoregressive_generate(target, [0], max_new_tokens=4, rng=Random(0))
        self.assertEqual(output, [0, 1, 0, 1, 0])
        self.assertEqual(stats.target_calls, 4)


if __name__ == "__main__":
    unittest.main()
