#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `stan2tfp` package."""


import unittest
from click.testing import CliRunner
import numpy as np
from stan2tfp import stan2tfp, sampling
from stan2tfp import cli
import pkg_resources


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

    def test_eight_schools_from_path(self):
        data_dict = dict(
            J=8, y=[28, 8, -3, 7, -1, 1, 18, 12], sigma=[15, 10, 16, 11, 9, 11, 10, 18]
        )
        model = stan2tfp.get_model_from_path(
            pkg_resources.resource_filename(__name__,"../tests/eight_schools_ncp.stan"),
            data_dict
        )
        mcmc_trace, _ = sampling.run_nuts(model)
        mu, tau, theta_tilde = [sampling.merge_chains(x) for x in mcmc_trace]

        self.assertAlmostEqual(significant_mean(mu), 4, delta=2)
        self.assertAlmostEqual(significant_mean(tau), 3, delta=2)
        self.assertAlmostEqual(significant_mean(theta_tilde), 0.08, delta=0.1)
    
    def test_eight_schools_from_stan_code(self):
        data_dict = dict(
            J=8, y=[28, 8, -3, 7, -1, 1, 18, 12], sigma=[15, 10, 16, 11, 9, 11, 10, 18]
        )
        stan_code = """
data {
  int<lower=0> J;
  real y[J];
  real<lower=0> sigma[J];
}

parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_tilde;
}

transformed parameters {
  vector[J] theta = mu + tau * theta_tilde;
}

model {
  mu ~ normal(0, 5);
  tau ~ normal(0, 5);
  theta_tilde ~ normal(0, 1);
  y ~ normal(theta, sigma);
}
        """
        model = stan2tfp.get_model_from_stan_code(
            stan_code,
            data_dict
        )
        mcmc_trace, _ = sampling.run_nuts(model)
        mu, tau, theta_tilde = [sampling.merge_chains(x) for x in mcmc_trace]

        self.assertAlmostEqual(significant_mean(mu), 4, delta=2)
        self.assertAlmostEqual(significant_mean(tau), 3, delta=2)
        self.assertAlmostEqual(significant_mean(theta_tilde), 0.08, delta=0.1)

if __name__ == "__main__":
    unittest.main()
