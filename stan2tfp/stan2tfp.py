# -*- coding: utf-8 -*-
from subprocess import Popen, PIPE
import os
import pkg_resources
import sys
import tempfile


def gen_tfp_code(path_to_stan_code):
    """call the stan2tfp executable and generate tfp code from stan code.
    
    Parameters
    ----------
    path_to_stan_code : string
        path to the .stan file to compile
    
    Returns
    -------
    string
        python program representing the same model with tfp
    
    Raises
    ------
    FileNotFoundError
        path to .stan file doesn't exists 
    """    
    plat = sys.platform
    if not os.path.exists(path_to_stan_code):
        raise FileNotFoundError(path_to_stan_code)
    call_stan2tfp_cmd = pkg_resources.resource_filename(
        __name__, "/bin/{}-stan2tfp.exe".format(plat)
    )
    stan2_tfp_input = path_to_stan_code
    cmd = [call_stan2tfp_cmd, stan2_tfp_input]
    proc = Popen(cmd, stdout=PIPE)
    tfp_code = proc.communicate()[0]
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
    """turn a tfp program and a data dictionary to a object nuts can sample
    
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