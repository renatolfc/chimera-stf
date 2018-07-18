#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
chimera - implementation of shared matrix factorization over time models
'''

from __future__ import print_function

import time
import logging
import scipy.io
import numpy as np
import tensorflow as tf

from sklearn.cluster import KMeans


def save_matrix(path, matrix):
    '''Saves a matrix to a file.

    Parameters
    ----------
        path : str
            Full path to the file in which the matrix will be saved.
        matrix : array-like
            The matrix to save
    '''
    scipy.io.mmwrite(path, matrix)


def load_matrix(path):
    '''Reads a matrix from a file in the Matrix Market format.

    Parameters
    ----------
        path : str
            Full path to the file from where the matrix is to be read.

    Returns
    -------
        m : array-like
            A matrix read from the file
    '''
    m = scipy.io.mmread(path)

    try:
        return m.todense()
    except AttributeError:
        return m


def load_array(path):
    '''Reads a label array as the ones used in this repo.

    This function performs no error handling.
    '''
    with open(path, 'rb') as fp:
        data = fp.read()
    return np.array([int(e) for e in data.split()]).reshape((-1, 1))


def save_array(path, array):
    '''Saves a label array (predicted or not) into path `path`.

    Parameters
    ----------
        path : str
            Full path to the output file
        array : array_like
            The array to save into de the file
    '''
    with open(path, 'wb') as fp:
        for a in array.reshape((-1,)):
            fp.write('%d ' % a)
        # remove last space
        fp.truncate(fp.tell() - 1)


def stf(A_t, C_t, K=10, steps=1000, alpha=0.001, beta=0.001, lambda1=0.005,
        lambda2=0.001, logdir=None, tolerance=1e-3):
    '''Factors matrices A_t and C_t into U, V_t, and W_t.

    This uses a somewhat naïve algorithm that holds everything in memory. It is
    easy to rewrite a version of the code that saves memory, but this is good
    enough for our purposes.

    Parameters
    ----------
        A_t : array_like
            An array, any object exposing the array interface. It is assumed to
            be a 3D ndarray with the time steps in the first dimension, the
            lines in the second, and the columns in the third. Must have the
            same first dimension of `C_t`.
        C_t : array_like
            An array, any object exposing the array interface. It is assumed to
            be a 3D ndarray with the time steps in the first dimension, the
            lines in the second, and the columns in the third. Must have the
            same first dimension of `A_t`.
        K : int
            Number of latend factors to use. Default: 10
        steps : int
            Number of epochs to run the gradient descent optimization. Default:
            1000
        alpha : float
            The learning rate of the gradient descent process. Default: 0.001
        beta : float
            The scaling factor that defines how much weight to give to the
            content matrix. Default: 0.001
        lambda1 : float
            Regularization constant for the weight matrices. Default: 0.005
        lambda2 : float
            Regularization constant for the temporal matrices. Default: 0.001
        logdir : str
            Path for saving tensorflow summary logs. Default: None
        tolerance : float
            Early stop optimization if loss ever goes below this value. Default:
            1e-3

    Returns
    --------
        out : tuple
            A tuple containing the final loss and the factorized matrices U_t
            (as a three-dimensional tensor), V, and W.
    '''
    tf.reset_default_graph()

    logging.info('Running Shared Matrix Factorization over time')

    A_t = np.asarray(A_t)
    C_t = np.asarray(C_t)

    if A_t.shape[0] != C_t.shape[0]:
        raise ValueError(
            'stf was called with matrices with different dimensions in the '
            'time axis.'
        )

    T = A_t.shape[0]
    N = A_t.shape[1]  # A's rows
    M = A_t.shape[-1]  # A's columns
    D = C_t.shape[-1]  # C's columns

    logging.debug(
        'k = %d, steps = %d, alpha = %g, beta = %g, lambda1 = %g, lambda2 = '
        '%g', K, steps, alpha, beta, lambda1, lambda2
    )

    # Variable definition {{{

    tUs = tf.Variable(
        tf.random_uniform(
            [T, N, K],
            maxval=1,
        ), name='U'
    )

    tV = tf.Variable(
        tf.random_uniform(
            [M, K],
            maxval=1,
        ), name='V_t'
    )

    tW = tf.Variable(
        tf.random_uniform(
            [D, K],
            maxval=1,
        ), name='W_t'
    )

    pAt = tf.placeholder(
        tf.float32,
        [T, N, M],
        name='A'
    )

    pCt = tf.placeholder(
        tf.float32,
        [T, N, D],
        name='C'
    )

    # Variable definition }}}

    lr = tf.placeholder(tf.float32, name='learning_rate')

    # Matrix multiplication {{{

    # When we first wrote this, TensorFlow did not support broadcasting for
    # matmul. I'm not sure it supports it now. What I know is that we can fake
    # broadcasting by forcing a couple of reshapes.
    tUs_ = tf.reshape(tUs, [T * N, K])
    tUs_tV = tf.matmul(tUs_, tf.transpose(tV))
    tUs_tW = tf.matmul(tUs_, tf.transpose(tW))

    # Matrix multiplication }}}

    # Error between approximations and true values {{{
    pAt_tUs_tV = pAt - tf.reshape(tUs_tV, [T, N, M])
    pCt_tUs_tW = pCt - tf.reshape(tUs_tW, [T, N, D])
    # Error between approximations and true values }}}

    # Losses {{{
    tA_loss = tf.reduce_sum(tf.square(pAt_tUs_tV))
    tC_tmp = tf.reduce_sum(tf.square(pCt_tUs_tW))
    tC_loss = beta * tf.reduce_sum(tC_tmp)
    tC_loss = tf.reduce_sum(tC_tmp)

    reg1 = lambda1 * (
        tf.reduce_sum(tf.square(tV)) +
        tf.reduce_sum(tf.square(tW)) +
        tf.reduce_sum(tf.square(tUs))
    )
    reg2 = lambda2 * tf.reduce_sum(tf.square(tUs[1:] - tUs[:T - 1]))

    loss = (tA_loss + tC_loss) + reg1 + reg2

    # Losses }}}

    # optimizer = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[tUs, tV, tW])
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list=[tUs, tV, tW])

    # Clip to yield non-negative factorization {{{

    # Clipping is not that necessary, but this is what the paper specifies
    tclip_U = tUs.assign(tf.maximum(0., tUs))
    tclip_V = tV.assign(tf.maximum(0., tV))
    tclip_W = tW.assign(tf.maximum(0., tW))
    clip = tf.group(tclip_U, tclip_V, tclip_W)

    # }}}

    if logdir:
        writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
    else:
        writer = None

    tf.summary.scalar("loss", loss)
    tf.summary.scalar('A_loss', tf.reduce_sum(tA_loss))
    tf.summary.scalar("C_loss", tf.reduce_sum(tC_loss))
    tf.summary.scalar("reg1", reg1)
    tf.summary.scalar("reg2", reg2)

    error = float('+inf')
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        for step in range(steps):
            error, summary, _ = sess.run(
                [loss, summary_op, optimizer],
                feed_dict={
                    pAt: A_t,
                    pCt: C_t,
                    lr: alpha,
                }
            )
            sess.run(clip)

            if writer:
                writer.add_summary(summary, step)

            if np.isinf(error) or np.isnan(error):
                raise Exception('optimization diverged')

            logging.debug('Current iteration: %d, current loss: %g',
                          step, error)

            if error < tolerance:
                break

        end_time = time.time()

        logging.info('Optimization ran for %g s and final loss is %g',
                     end_time - start_time, error)

        Us = sess.run(tUs)
        V = sess.run(tV)
        W = sess.run(tW)

    return error, Us, V, W


def community_detection(Us, n_communities, **kwargs):
    '''Detects communities in the U matrices (latent representation of A & C).

    This community detection code follows the process described in the paper:
    essentially, we use the dense embedding / latent representation of A_t and
    C_t to perform k-means and find communities.

    Parameters
    ----------
        Us : array_like
            An array, any object exposing the array interface. It is assumed to
            be a 3D ndarray with the time steps in the first dimension, the
            lines in the second, and the columns in the third. This is expected
            to be the same array output by `stf`.
        n_clusters : int
            The number of communities (clusters in K-Means) to search for.
        kwargs : dict
            Keyword arguments to `sklearn.cluster.KMeans`

    Returns
    --------
        out : array_like
            A matrix containing all the labels of the lines of U over time.
            If Us contains three time-steps and 1000 entries, the output
            will be an ndarray with this shape (plus one column).
    '''

    cache = {}

    def mapping(x):
        if x not in cache:
            cache[x] = len(cache)
        return cache[x]

    kmeans = KMeans(n_clusters=n_communities, **kwargs)
    labels = kmeans.fit_predict(Us.reshape((-1, Us.shape[-1])))
    return np.array([mapping(l) for l in labels])
