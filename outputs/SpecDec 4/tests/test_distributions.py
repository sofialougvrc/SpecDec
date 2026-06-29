from __future__ import annotations

import unittest

from specdec.distributions import (
    acceptance_probability,
    positive_difference_distribution,
)


class DistributionTests(unittest.TestCase):
    def test_acceptance_probability_caps_ratio(self) -> None:
        self.assertEqual(acceptance_probability(0.8, 0.2), 1.0)
        self.assertAlmostEqual(acceptance_probability(0.2, 0.8), 0.25)

    def test_positive_difference_distribution(self) -> None:
        adjusted = positive_difference_distribution(
            target=[0.1, 0.5, 0.4],
            draft=[0.2, 0.2, 0.6],
        )
        self.assertEqual(adjusted[0], 0.0)
        self.assertAlmostEqual(adjusted[1], 1.0)
        self.assertEqual(adjusted[2], 0.0)


if __name__ == "__main__":
    unittest.main()
