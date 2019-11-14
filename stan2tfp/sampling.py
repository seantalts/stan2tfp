import tensorflow as tf
import tensorflow_probability as tfp

# from pprint import pprint
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors
dtype = tf.float64


def _step_size_setter_fn(pkr, new_step_size):
    return pkr._replace(
        inner_results=pkr.inner_results._replace(step_size=new_step_size)
    )


@tf.function(experimental_compile=True)
def run_nuts(model, nchain=4, num_main_iters=1000, num_warmup_iters=1000):
    """Draw samples from the model using NUTS.
    
    :param model: A tfp object representing a model to sample from
    :type model: tfd.Distribution
    :param nchain: Number of chains to sample, defaults to 4
    :type nchain: int, optional
    :param num_main_iters: The number of samples to draw, defaults to 1000
    :type num_main_iters: int, optional
    :param num_warmup_iters: The number of warmup iterations, defaults to 1000
    :type num_warmup_iters: int, optional
    :return: Tuple of two elements:
        1. mcmc_trace - a list samples drawn from the model
        2. pkr (previous kernel results) - a dictionary of sampler statistics defined by trace_fn 
    :rtype: tuple
    """
    initial_states = [
        tf.random.uniform(s, -2, 2, dtype, name="initializer")
        for s in model.parameter_shapes(nchain)
    ]
    step_sizes = [1e-2 * tf.ones_like(i) for i in initial_states]
    kernel = tfp.mcmc.TransformedTransitionKernel(
        tfp.mcmc.nuts.NoUTurnSampler(
            target_log_prob_fn=lambda *args: model.log_prob(args), step_size=step_sizes
        ),
        bijector=model.parameter_bijectors(),
    )

    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel,
        target_accept_prob=tf.cast(0.8, dtype=dtype),
        # Adapt for the entirety of the trajectory.
        num_adaptation_steps=num_warmup_iters,
        step_size_setter_fn=_step_size_setter_fn,
        step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
    )

    # Sampling from the chain.
    mcmc_trace, pkr = tfp.mcmc.sample_chain(
        num_results=num_main_iters,
        num_burnin_steps=num_warmup_iters,
        current_state=[
            bijector.forward(state)
            for bijector, state in zip(model.parameter_bijectors(), initial_states)
        ],
        kernel=kernel,
    )

    return mcmc_trace, pkr


def merge_chains(a):
    """merge samples from different chains to a single numpy array
    
    :param a: samples, shape (n_chains, n_iter, ...)
    :type a: Tensor
    :return: samples, shape (n_chains * n_iter, ...)
    :rtype: ndarray
    """    
    return np.reshape(a, a.shape[0] * a.shape[1] + a.shape[2:])
