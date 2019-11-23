========
stan2tfp
========


.. .. image:: https://img.shields.io/pypi/v/stan2tfp.svg
..         :target: https://pypi.python.org/pypi/stan2tfp

.. image:: https://img.shields.io/travis/adamhaber/stan2tfp.svg
        :target: https://travis-ci.org/adamhaber/stan2tfp

.. image:: https://readthedocs.org/projects/stan2tfp/badge/?version=latest
        :target: https://stan2tfp.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

The new Stan compiler features a TensorFlow Probability backend, translating Stan programs to python code. 
stan2tfp is a lightweight python wrapper around this functionality, allowing users to:


* call the compiler (emitting TFP code)
* run the code (creating a model object in the current namespace)
* sample the model (using TFP's NUTS)

without leaving the notebook or their favorite IDE.

The new Stan compiler and the TFP backend are under active development. Currently only a small subset of Stan's functionality is supported. For a list of supported distributions, see ...

Installation
============

...


Example
=======

...

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
