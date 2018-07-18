#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import logging

import numpy as np
from sklearn.metrics import jaccard_similarity_score

from .context import chimera, datapath

COMMUNITIES = 5
LATENT_FACTORS = 7
DATASET = 'synthetic'
SYNTHETIC_LINKS = 5000
LABELS = list(sorted(glob.glob(os.path.join(datapath, DATASET, 'labels*'))))
A_MATRICES = list(sorted(glob.glob(os.path.join(datapath, DATASET, 'A?.mtx*'))))
C_MATRICES = list(sorted(glob.glob(os.path.join(datapath, DATASET, 'C?.mtx*'))))

assert len(A_MATRICES)
assert len(C_MATRICES)
assert len(LABELS)

assert len(A_MATRICES) == len(C_MATRICES) == len(LABELS)


def test_load_matrix():
    m = chimera.load_matrix(A_MATRICES[0])
    assert len(m.shape) == 2
    assert m.shape == (SYNTHETIC_LINKS, SYNTHETIC_LINKS)


def test_load_array():
    a = chimera.load_array(LABELS[0])
    assert len(a.shape) == 2
    assert a.shape == (SYNTHETIC_LINKS, 1)


def test_e2e():
    As = np.array([chimera.load_matrix(m) for m in A_MATRICES])
    Cs = np.array([chimera.load_matrix(m) for m in C_MATRICES])
    y = np.array([chimera.load_array(a) for a in LABELS])

    print('As', As)
    print('Cs', Cs)
    print('y', y)

    assert As.shape[0] == Cs.shape[0] == y.shape[0]

    loss, Us, V, W = chimera.stf(
        As,
        Cs,
        LATENT_FACTORS,
        alpha=0.0001,
        beta=1000,
        lambda1=0.1,
        lambda2=0.0001,
        steps=500,
    )

    # These should be matrices
    assert len(V.shape) == len(W.shape)

    logging.info('Final optimization loss (for 500 steps): %g', loss)

    yhat = chimera.community_detection(
        Us,
        COMMUNITIES,
        init='k-means++',
        n_init=100,
        max_iter=3000,
        tol=0.0001,
    ).reshape((As.shape[0], As.shape[1], 1))

    logging.info('Final score: %g', jaccard_similarity_score(y.reshape((-1,)), yhat.reshape(-1,)))

    chimera.save_array('labels.txt', yhat)
