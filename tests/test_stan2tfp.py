#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `stan2tfp` package."""


import unittest
from click.testing import CliRunner
import numpy as np
from stan2tfp import stan2tfp, sampling
from stan2tfp import cli


def significant_digit(x):
    return float("{:.1}".format(x))


def significant_mean(x):
    return significant_digit(np.mean(x))


class TestStan2tfp(unittest.TestCase):
    """Tests for `stan2tfp` package."""

    # def setUp(self):
    #     """Set up test fixtures, if any."""

    # def tearDown(self):
    #     """Tear down test fixtures, if any."""

    def test_eight_schools(self):
        data_dict = dict(
            J=8, y=[28, 8, -3, 7, -1, 1, 18, 12], sigma=[15, 10, 16, 11, 9, 11, 10, 18]
        )
        model = stan2tfp.get_model_from_path(
            "/Users/adamhaber/projects/stan2tfp/tests/eight_schools_ncp.stan", data_dict
        )
        mcmc_trace, _ = sampling.run_nuts(model)
        mu, tau, theta_tilde = [sampling.merge_chains(x) for x in mcmc_trace]

        self.assertAlmostEqual(significant_mean(mu), 4, delta=2)
        self.assertAlmostEqual(significant_mean(tau), 3, delta=2)
        self.assertAlmostEqual(significant_mean(theta_tilde), 0.08, delta=0.1)

    # def test_command_line_interface(self):
    #     """Test the CLI."""
    #     runner = CliRunner()
    #     result = runner.invoke(cli.main)
    #     assert result.exit_code == 0
    #     assert 'stan2tfp.cli.main' in result.output
    #     help_result = runner.invoke(cli.main, ['--help'])
    #     assert help_result.exit_code == 0
    #     assert '--help  Show this message and exit.' in help_result.output


if __name__ == "__main__":
    unittest.main()
