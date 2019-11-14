# -*- coding: utf-8 -*-
from subprocess import Popen, PIPE
import os
import pkg_resources

def gen_code(path_to_stan_code):
    call_stan2tfp_cmd = pkg_resources.resource_filename(__name__, "/bin/stan2tfp.exe")
    stan2_tfp_input = path_to_stan_code
    cmd = [call_stan2tfp_cmd, stan2_tfp_input]
    proc = Popen(cmd, stdout=PIPE)
    tfp_code = proc.communicate()[0]
    return tfp_code


def save_code(tfp_code, fname):
    with open(fname, "w") as f:
        f.writelines(tfp_code.split("\n"))


def get_model_obj(tfp_code):
    exec_dict = {}
    exec(tfp_code, exec_dict)
    return exec_dict["model"]


def init_model_with_data(model_obj, data_dict):
    return model_obj(**data_dict)


def get_model_from_code(tfp_code, data_dict):
    model_obj = get_model_obj(tfp_code)
    model = init_model_with_data(model_obj, data_dict)
    return model


def get_model_from_path(path_to_stan_code, data_dict):
    tfp_code = gen_code(path_to_stan_code)
    model_obj = get_model_obj(tfp_code)
    model = init_model_with_data(model_obj, data_dict)
    return model


# """Main module."""
