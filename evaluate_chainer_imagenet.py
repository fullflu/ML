#!/usr/bin/env python
#coding:utf-8

from __future__ import print_function
import argparse
import math
import sys
import time

import numpy as np
import six

import chainer
from chainer import cuda
#import chainer.links as L
from chainer import optimizers
from chainer import serializers

from PIL import Image
import os
import datetime
import json
import multiprocessing
import random
import threading
import six.moves.cPickle as pickle
from six.moves import queue
from chainer import computational_graph

parser = argparse.ArgumentParser()
parser.add_argument('test', help='Path to test image-label list file')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--initmodel', '-init', default='model',
                    help='Initialize the model from given file')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of image files')
parser.add_argument('--arch', '-a', default='nin',
                    help='Convnet architecture \
                    (nin, alex, alexbn, googlenet, googlenetbn)')
parser.add_argument('--test_batchsize', '-b', type=int, default=1,
                    help='test minibatch size, you should not change this!!')
parser.add_argument('--mean', '-m', default='mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--loaderjob', '-j', default=20, type=int,
                    help='Number of parallel data loading processes')

args = parser.parse_args()

xp = cuda.cupy if args.gpu >= 0 else np

# Prepare model
if args.arch == 'nin':
    import nin
    model = nin.NIN()
elif args.arch == 'alex':
    import alex
    model = alex.Alex()
elif args.arch == 'alexbn':
    import alexbn
    model = alexbn.AlexBN()
elif args.arch == 'googlenet':
    import googlenet
    model = googlenet.GoogLeNet()
elif args.arch == 'googlenetbn':
    import googlenetbn
    model = googlenetbn.GoogLeNetBN()
else:
    raise ValueError('Invalid architecture name')

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
else:
	print('cannot evaluate model!!')

def load_image_list(path, root):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append((os.path.join(root, pair[0]), np.int32(pair[1])))
    return tuples

test_list = load_image_list(args.test, args.root)
mean_image = pickle.load(open(args.mean, 'rb'))
len_test = len(test_list)

data_q = queue.Queue(maxsize=1)
res_q = queue.Queue()

cropwidth = 256 - model.insize

def read_image(path, center=False, flip=False):
	# for simple input
    # Data loading routine
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)
    if center:
        top = left = cropwidth / 2
    else:
        top = random.randint(0, cropwidth - 1)
        left = random.randint(0, cropwidth - 1)
    bottom = model.insize + top
    right = model.insize + left

    image = image[:, top:bottom, left:right].astype(np.float32)
    image -= mean_image[:, top:bottom, left:right]
    image /= 255
    if flip and random.randint(0, 1) == 0:
        return image[:, :, ::-1]
    else:
        return image


def feed_data():
    # Data feeder
    i = 0
    count = 0

    test_x_batch = np.ndarray(
        (args.test_batchsize, 3, model.insize, model.insize), dtype=np.float32)
    test_y_batch = np.ndarray((args.test_batchsize,), dtype=np.int32)

    test_batch_pool = [None] * args.test_batchsize
    pool = multiprocessing.Pool(args.loaderjob)
    data_q.put("test")
    j = 0
    for path,label in test_list:
    	test_batch_pool[j] = pool.apply_async(read_image,(path,True,False))
    	test_y_batch[j] = label
    	j += 1

    	if j == args.test_batchsize:
    		for k, x in enumerate(test_batch_pool):
    			test_x_batch[k] = x.get()
    		data_q.put((test_x_batch.copy(), test_y_batch.copy()))
    		j = 0
    pool.close()
    pool.join()
    data_q.put("end")
    pool.close()
    pool.join()
    data_q.put('end')

def log_result():
    # Logger
    begin_at = time.time()
    test_begin_at = None
    while True:
        result = res_q.get()
        if result == 'end':
            print(file=sys.stderr)
            break
        elif result == 'test':
            print(file=sys.stderr)
            train = False
            test_count = test_loss = test_accuracy = 0
            test_begin_at = time.time()
            continue
            
        loss, accuracy = result

        test_count += args.test_batchsize
        duration = time.time() - test_begin_at
        throughput = test_count / duration
        #sys.stderr.write(
        #print(
            #'\rval   {} batches ({} samples) time: {} ({} images/sec)'
            #.format(test_count / args.test_batchsize, test_count,
                    #datetime.timedelta(seconds=duration), throughput))

        test_loss += loss
        test_accuracy += accuracy
        if test_count == len_test:#000:
            mean_loss = test_loss * args.test_batchsize / len_test
            mean_error = 1 - test_accuracy * args.test_batchsize / len_test
            print(file=sys.stderr)
            print(json.dumps({'type': 'test', 'iteration': test_count,
                              'error': mean_error, 'loss': mean_loss}))
            sys.stdout.flush()


def test_net():
    while True:
        while data_q.empty():
            time.sleep(0.1)
        inp = data_q.get()
        if inp == 'end':  # quit
            res_q.put('end')
            break
        elif inp == 'test':  # restart training
            res_q.put('test')
            model.train = False
            continue

        volatile = 'off' if model.train else 'on'
        x = chainer.Variable(xp.asarray(inp[0]), volatile=volatile)
        t = chainer.Variable(xp.asarray(inp[1]), volatile=volatile)

        model(x,t)
        res_q.put((float(model.loss.data), float(model.accuracy.data)))
        del x, t

# Invoke threads
feeder = threading.Thread(target=feed_data)
feeder.daemon = True
feeder.start()
logger = threading.Thread(target=log_result)
logger.daemon = True
logger.start()

test_net()
feeder.join()
logger.join()

