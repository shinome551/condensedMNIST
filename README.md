# condensedMNIST

## Abstract
MNIST trainset can be reduced from 60000 to 20953 by using a Relative Neighborhood Graph(RNG). The classification results in DNN or kNN are no different from the all trainset and better than random sampling.

## Method
RNG is one of the neighbor graphs. It's represent adjacency between nodes as their edges. 

When creating a RNG for a dataset, take the following steps:
1. For a pair of examples, draw two hypersphere whose radius is the distance between the pairs.
2. If no other examples exist in the overlapping region of the two hyperspheres, connect a edge.

After creation, remove the examples not connected to ones belonging to different classes. Thereby, we get the representative subset for decision boundaries.

## Results
### DNN Classification(dnn.{py, ipynb})
The test accuracy for each dataset when training DNN with 4 fully connected layers and ReLU. 

The number of epochs is 100 but the number of iterations is kept the same.

| Mode          | Number of Samples |  Test Acc[%] |
|---------------|:-----------------:|:------------:|
| All data      |       60000       |     97.8     |
| Condensed     |       20953       |     98.0     |
| Random        |       20953       |     96.7     |

### k-Nearest-Neighbor Classification(kNN.{py, ipynb})
- k=1, L2 distance

| Mode          | Number of Samples |  Test Acc[%] |
|---------------|:-----------------:|:------------:|
| All data      |       60000       |     96.3     |
| Condensed     |       20953       |     95.7     |
| Random        |       20953       |     95.0     |
