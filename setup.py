#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.core import setup
from distutils.extension import Extension

import chimera


setup(
    name='chimera-stf',
    version=chimera.__version__,
    description='Chimera Shared Matrix Factorization over Time',
    author='Renato L. F. Cunha',
    url='https://github.com/renatolfc/chimera-stf',
    packages=['chimera'],
    setup_requires=[
        'setuptools>=18.0',
    ],
    install_requires=[
        'scikit-learn',
        'numpy',
        'pandas',
        'tensorflow',
        'scipy',
    ],
)
