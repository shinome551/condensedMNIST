#!/usr/bin/env python
# coding: utf-8

import argparse

import numpy as np
from torchvision.datasets import MNIST
import jax
import jax.numpy as jnp
from jax import jit, vmap

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, help='Use this option in \'condensed\' mode only.')
    parser.add_argument('--index', type=str)
    parser.add_argument('--num_samples', type=int, default=20953, help='Choice \'full\', \'condensed\', or \'random\'')

    
    args = parser.parse_args()

    trainset = MNIST(root='./data', train=True, download=True)
    testset = MNIST(root='./data', train=False, download=True)

    train_data = trainset.data.flatten(1).numpy() / 255.0
    train_data = jax.device_put(train_data)
    train_targets = jnp.array(trainset.targets)

    if args.mode == 'condensed':
        print('use condensed subset')
        if args.index is not None:
            index_condensed = np.loadtxt(args.index, dtype=int)
        else:
            raise ValueError('invalid index path')        
        train_data = train_data[index_condensed]
        train_targets = train_targets[index_condensed]
    elif args.mode == 'random':
        print('use random dataset')
        rng = np.random.default_rng(args.seed)
        index_random = np.arange(len(trainset))
        rng.shuffle(index_random)
        index_random = index_random[:len(args.num_samples)]
        train_data = train_data[index_random]
        train_targets = train_targets[index_random]
    else:
        print('use full dataset')

    test_data = testset.data.flatten(1).numpy() / 255.0
    test_data = jax.device_put(test_data)
    test_targets = jnp.array(testset.targets)

    num_classes = len(jnp.unique(test_targets))

    def l2dist(x, y):
        return jnp.linalg.norm(x - y, ord=1)

    mv_l2dist = vmap(l2dist, (0, None), 0)

    @jit
    def searchkNN(train_data, train_targets, test_data, test_targets, idx, top_k):
        query, query_target = test_data[idx], test_targets[idx]
        v_dist = mv_l2dist(train_data, query)
        indices = jnp.take(jnp.argsort(v_dist), top_k)
        predict = jnp.bincount(train_targets[indices], length=num_classes).argmax()
        return query_target == predict

    correct = 0
    top_k = jnp.arange(1)
    for idx in range(len(testset)):
        correct += searchkNN(
            train_data, 
            train_targets, 
            test_data, 
            test_targets, 
            idx,
            top_k
        )
    accuracy = 100 * correct.item() / len(testset)
    print(f'Test Accuracy:{accuracy:.1f}%')

        
if __name__ == '__main__':
    main()