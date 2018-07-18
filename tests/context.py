#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, '..')))

import chimera

datapath = os.path.abspath(os.path.join(here, '..', 'data'))
