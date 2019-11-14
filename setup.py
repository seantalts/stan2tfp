#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

# requirements = [
#     "Click>=7.0",
# ]

# with open('requirements.txt') as f:
requirements = [
    "pip",
    "numpy",
    "gast==0.2.2",
    "tf-nightly-2.0-preview",
    "tfp-nightly",
    "bump2version==0.5.11",
    "wheel==0.33.6",
    "watchdog==0.9.0",
    "flake8==3.7.8",
    "tox==3.14.0",
    "coverage==4.5.4",
    "Sphinx==1.8.5",
    "twine==1.14.0",
    "Click==7.0",
]

setup_requirements = []

test_requirements = []

setup(
    author="Adam Haber",
    author_email="adamhaber@gmail.com",
    python_requires="!=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Python Boilerplate contains all the boilerplate you need to create a Python package.",
    entry_points={"console_scripts": ["stan2tfp=stan2tfp.cli:main",],},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="stan2tfp",
    name="stan2tfp",
    packages=find_packages(include=["stan2tfp", "stan2tfp.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/adamhaber/stan2tfp",
    version="0.1.0",
    zip_safe=False,
)
