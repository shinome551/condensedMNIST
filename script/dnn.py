#!/usr/bin/env python
# coding: utf-8

import argparse

import numpy as np
from numpy.random import default_rng
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose


def initSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def repeatIndex(index, target_num):
    ite = target_num // len(index)
    index = np.tile(index, ite + 1)[:target_num]
    return index


class Trainer:
    def __init__(self, model, trainset, testset, cfg):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_epochs = cfg['num_epochs']
        self.model = model.to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=cfg['lr'], momentum=0.9, nesterov=True, weight_decay=1e-4)
        batch_size = cfg['batch_size']
        self.trainloader = DataLoader(trainset, batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=2, 
                                pin_memory=True)
        self.testloader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True)


    def train(self):
        self.model.train()
        trainloss = 0
        for data in self.trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            self.optimizer.step()
            trainloss += loss.item() * inputs.size()[0]

        trainloss = trainloss / len(self.trainloader.dataset)
        return trainloss


    def test(self):
        self.model.eval()
        testacc = 0
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                testacc += (predicted == labels).sum().item()

        testacc = 100 * testacc / len(self.testloader.dataset)
        return testacc


    def run(self):
        for epoch in range(self.num_epochs):
            trainloss = self.train()
            testacc = self.test()
            print(f'epoch:{epoch+1}, trainloss:{trainloss:.3f}, testacc:{testacc:.1f}%')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--index', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--num_samples', type=int, default=20953)
    parser.add_argument('--repeat', action='store_false')
    args = parser.parse_args()

    initSeed(args.seed)

    cfg = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'momentum':args.momentum,
        'weight_decay': args.weight_decay
    }
    print('config:', cfg)

    transform = Compose([
        ToTensor(),
        Normalize(mean=0.5, std=0.5)
    ])
    
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    if args.mode == 'condensed':
        print('use condensed subset')
        if args.index is not None:
            index_condensed = np.loadtxt(args.index, dtype=int)
        else:
            raise ValueError
        if args.repeat:
            index_condensed = repeatIndex(index_condensed, len(trainset))
        trainset = Subset(trainset, index_condensed)
    elif args.mode == 'random':
        print('use random subset')
        rng = default_rng(args.seed)
        index_random = np.arange(len(trainset))
        rng.shuffle(index_random)
        index_random = index_random[:args.num_samples]
        if args.repeat:
            index_random = repeatIndex(index_random, len(trainset))
        trainset = Subset(trainset, index_random)
    else:
        print('use all dataset')
    
    print(f'num_samples:{len(trainset)}')

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )

    print('start training')
    trainer = Trainer(model, trainset, testset, cfg)
    trainer.run()
    print('\ntrain finished!')

    
if __name__ == '__main__':
    main()