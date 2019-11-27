# -*- coding: utf-8 -*-
from subprocess import run, PIPE
import os
import pkg_resources
import sys
import tempfile

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors
dtype = tf.float64

class Stan2tfp():
    
    slots = ['compiler_path','tfp_code','stan_model_code','parameter_shapes','parameter_bijectors', 'model','model_constructor']

    def __init__(self, stan_file_path=None, stan_model_code=None, data_dict=None):
        """Construct a TensorFlow Probability model from a Stan model.
        
        Parameters
        ----------
        stan_file_path : string, optional, by default None
            Model code must found via one of the following parameters: `stan_file_path` or `stan_model_code`.
            The string passed as an argument is expected to be a filename containing the Stan model specification.
    
        stan_model_code : string, optional
            A string containing the Stan model specification.

        data_dict : dict, optional
            A Python dictionary providing the data for the model. Variables for Stan are stored in the
            dictionary as expected. Variable names are the keys and the values are their associated values.
            
            Data for the model can be provided later using the `init_model` function,
            but must be provided before sampling.
        
        Raises
        ------
        ValueError
            If both stan_file_path and stan_model_code are None.
        FileNotFoundError
            If stan_file_path is not a valid path.
        """        
        super().__init__()
        self.parameter_shapes = None
        self.parameter_bijectors = None
        self._set_compiler_path()

        # call the compiler
        if stan_file_path is None:
            if stan_model_code is None:
                raise ValueError("Either stan_model_code or stan_file_path must be provided to create a Model object")
            else:
                self.stan_model_code = stan_model_code
                self.tfp_code = self._tfp_from_stan_model_code(stan_model_code)
        else:
            if not os.path.exists(stan_file_path):
                raise FileNotFoundError(stan_file_path)
            with open(stan_file_path, 'r') as f:
                self.stan_model_code = f.read()
            self.tfp_code = self._tfp_from_stan_file(stan_file_path)
        
        # execute tfp_code in the current namespace
        exec_dict = {}
        exec(self.tfp_code, exec_dict)
        self.model_constructor = exec_dict["model"]

        if data_dict is not None:
            self.model = self.model_constructor(**data_dict)
        else:
            self.model = None

    def init_model(self, data_dict):
        """Instantiate a TFP model with data. Initialization is required for sampling.  
        
        Parameters
        ----------
        data_dict : dict
            A Python dictionary providing the data for the model. Variables for Stan are stored in the
            dictionary as expected. Variable names are the keys and the values are their associated values.

            If data has been passed previously (by the constructor or the init_model function),
            it will be overwritten. This is useful for calling the same model with different data.
        """        
        self.model = self.model_constructor(**data_dict)
        self.parameter_bijectors = self.model.parameter_bijectors()
        self.parameter_shapes = self.model.parameter_shapes(1)

    @tf.function(experimental_compile=True)
    def sample(self, nchain=4, num_main_iters=1000, num_warmup_iters=1000):
        """Draw samples from the model using NUTS.
        
        Parameters
        ----------
        nchain : int, optional
            Positive integer specifying number of chains, 4 by default.
        num_main_iters : int, optional
            Positive integer specifying how many iterations for each chain after warmup, 1000 by default.
        num_warmup_iters : int, optional
            Positive integer specifying number of warmup (aka burin) iterations.
            As `warmup` also specifies the number of iterations used for stepsize
            adaption, warmup samples should not be used for inference. 1000 by default.
        
        Returns
        -------
        (mcmc_trace, pkr) : tuple
            mcmc_trace - a list samples drawn from the model
            pkr (previous kernel results) - a dictionary of sampler statistics as defined by trace_fn 
        """      
        if self.model is None:
            return ValueError("The model class has not been instantiated. Call init_model with the the observed data.")

        initial_states = [
            tf.random.uniform(s, -2, 2, dtype, name="initializer")
            for s in self.model.parameter_shapes(nchain)
        ]
        step_sizes = [1e-2 * tf.ones_like(i) for i in initial_states]
        kernel = tfp.mcmc.TransformedTransitionKernel(
            tfp.mcmc.nuts.NoUTurnSampler(
                target_log_prob_fn=lambda *args: self.model.log_prob(args), step_size=step_sizes
            ),
            bijector=self.model.parameter_bijectors(),
        )

        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            kernel,
            target_accept_prob=tf.cast(0.8, dtype=dtype),
            # Adapt for the entirety of the trajectory.
            num_adaptation_steps=num_warmup_iters,
            step_size_setter_fn=self._step_size_setter_fn,
            step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
        )

        # Sampling from the chain.
        mcmc_trace, pkr = tfp.mcmc.sample_chain(
            num_results=num_main_iters,
            num_burnin_steps=num_warmup_iters,
            current_state=[
                bijector.forward(state)
                for bijector, state in zip(self.model.parameter_bijectors(), initial_states)
            ],
            kernel=kernel,
        )

        return mcmc_trace, pkr
    
    def merge_chains(self, a):
        """Merge samples from different chains to a single numpy array
        
        Parameters
        ----------
        a : tensor
            samples, shape (n_chains, n_iter, ...)
        
        Returns
        -------
        ndarray
            samples, shape (n_chains * n_iter, ...)
        """        
        return np.reshape(a, a.shape[0] * a.shape[1] + a.shape[2:])

    def get_tfp_code(self):
        """Returns the string representation of the TFP model.
        
        Returns
        -------
        string
            the TFP model
        """        
        return self.tfp_code.decode("UTF-8")

    def save_tfp_code(self, fname):
        """Save the tfp model to file
        
        Parameters
        ----------
        fname : string
            output file in which to save the model
        """        
        with open(fname, "w") as f:
            f.writelines(self.get_tfp_code())

    def _set_compiler_path(self):
        plat = sys.platform
        if plat not in ['darwin', 'linux', 'win32']:
            raise OSError("OS {} is not supported".format(plat))
        self.compiler_path = pkg_resources.resource_filename(
            __name__, "/bin/{}-stan2tfp.exe".format(plat)
        )

    def _tfp_from_stan_file(self, stan_file_path):
        cmd = [self.compiler_path, stan_file_path]
        print("Compiling stan file to tfp file...")
        proc = run(cmd, stdout=PIPE, stderr=PIPE)
        tfp_code = proc.stdout
        return tfp_code
    
    def _tfp_from_stan_model_code(self, stan_model_code):
        fd, path = tempfile.mkstemp(prefix="name",suffix="asda")
        with open(fd, "w") as f:
            f.write(stan_model_code)
        tfp_code = self._tfp_from_stan_file(path)
        os.unlink(path)
        return tfp_code

    def _step_size_setter_fn(self, pkr, new_step_size):
        return pkr._replace(
            inner_results=pkr.inner_results._replace(step_size=new_step_size)
        )