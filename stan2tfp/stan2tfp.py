# -*- coding: utf-8 -*-
from subprocess import Popen, PIPE
import os
import pkg_resources
import sys
import tempfile


def generate_tfp_code(stan_code_path):
    """call the stan2tfp executable and generate tfp code from stan code.
    
    Parameters
    ----------
    stan_code_path : string
        path to the .stan file to compile
    
    Returns
    -------
    string
        textual python code representing the same stan model with tfp
    
    Raises
    ------
    FileNotFoundError
        path to .stan file doesn't exists 
    """
    plat = sys.platform
    if not os.path.exists(stan_code_path):
        raise FileNotFoundError(path_to_stan_code)
        
    call_stan2tfp_cmd = pkg_resources.resource_filename(
        __name__, "/bin/{}-stan2tfp.exe".format(plat)
    ) # TOMER: I didn't fully understand what is this used for, but: 1. does 'exe' format will run on linux? 2. what happens if the user's plattform is not supported? I think you want to handle this error, no?
    stan2_tfp_input = stan_code_path # TOMER: Why is this necessary?
    cmd = [call_stan2tfp_cmd, stan2_tfp_input]
    proc = Popen(cmd, stdout=PIPE)
    tfp_code = proc.communicate()[0] # TOMER: Consider using subprocess.run() or subprocess.check_output(), it seems like these are more updated functions that will save you some lines of code and handle errors better (I think). look here: https://docs.python.org/3/library/subprocess.html#subprocess.run (they also talk there about security considerations)
    
    return tfp_code


def save_code(tfp_code, fname):
    """helper function for saving the generated python program
    
    Parameters
    ----------
    tfp_code : string
        the program to save
    fname : string
        path in which to save the program
    
    """
    with open(fname, "w") as f:
        f.writelines(tfp_code.decode("UTF-8").split("\n"))


def get_tfp_model_obj(tfp_code):
    """Executes the generated python program, and returns an
    un-instantiated model object
    
    Parameters
    ----------
    tfp_code : string
        the program to execute
    
    Returns
    -------
    tfd.Distribution
        the model class
    """    
    exec_dict = {}
    exec(tfp_code, exec_dict)
    return exec_dict["model"]


def init_model_with_data(model_obj, data_dict):
    """instantiate the generated tfp model object with data
    
    Parameters
    ----------
    model_obj : tfd.Distribution
        an un-instantiated model object
    data_dict : dict
        data for the model
    
    Returns
    -------
    tfd.Distribution
        an instantiated model with all the methods necessary for sampling
    """    
    return model_obj(**data_dict)


def model_from_tfp_code(tfp_code, data_dict):
    """turn a tfp program and a data dictionary to an object nuts can sample
    
    Parameters
    ----------
    tfp_code : string
        the tfp program 
    data_dict : dictionary
        data for the model
    
    Returns
    -------
    tfd.Distribution
        an instantiated model with all the methods necessary for sampling
    """    
    tfp_model_obj = get_tfp_model_obj(tfp_code)
    model = init_model_with_data(tfp_model_obj, data_dict)
    return model


def model_from_path(path_to_stan_code, data_dict):
    """create a tfp object nuts can sample from a .stan file and a data dictionary
    
    Parameters
    ----------
    path_to_stan_code : string
        path to the .stan file to compile
    data_dict : dictionary
        data for the model
    
    Returns
    -------
    tfd.Distribution
        an instantiated model with all the methods necessary for sampling
    """    
    tfp_code = gen_tfp_code(path_to_stan_code)
    model = model_from_tfp_code(tfp_code, data_dict)
    return model


def model_from_stan_code(stan_code, data_dict):
    """create a tfp object nuts can sample from a stan code and a data dictionary
    
    Parameters
    ----------
    stan_code : string
        multi-line string representing a stan program
    data_dict : dictionary
        data for the model
    
    Returns
    -------
    tfd.Distribution
        an instantiated model with all the methods necessary for sampling
    """    
    fd, path = tempfile.mkstemp()
    with open(fd, "w") as f:
        f.write(stan_code)
    model = model_from_path(path, data_dict)
    os.unlink(path)
    return model

'''
TOMER:
1. Why have you decided to expose so many functions to the user ("public functions")? In general, I think that you over-divided your
   "not too long" code into too much functions. From my point of view, this module should export only 1 function, something like "stan2tfp"
   that recieves 1. either a stan_file_path or a stan_code (both strings that you can check first whether to read the file or continue
   with the textual code) 2. the data dict, and returns an instantiated tfp model object.
   This function will call to the compiler, execute the generated python code, assign the data and return. (few of these sub-operations
   can be implemented by private small functions).
2. Actually, better than that, and if you have power, I think that I would create a class. It will be initialized by stan_path or stan_code.
   It will hold as attributes the stan_path, stan_code, tfp_code, the uninstantiated model. It will have methods for instantiating the
   tfp model with different data; saving the tfp_code to a file; and even, after quick glancing on your sampling.py, incorporate these
   features here (sample from the model, etc)
'''
