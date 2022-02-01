# condensedMNIST
By using a Relative Neighborhood Graph (RNG), MNIST trainset can be reduced from 60000 to 20953. Classification results using the trainset in DNN and kNN are no different from the all trainset and better than random sampling.

## Method
RNG is one of the neighbor graphs. It's represent adjacency between nodes as their edges. 

When make a RNG for a dataset, take the following steps:
1. For each pair of examples, draw two hyperspheres centered on each one of the pair, with a radius being the Euclidean distance between the pair(as shown in the image below).
2. If no other examples exist in the overlapping region of the two hyperspheres, connect an edge between the pair.

<div align="center">
    <img src="img/figure1.png" width="40%">
</div>

After making, remove the examples not connected to ones belonging to different classes. Thereby, we get the representative subset for decision boundaries. Such data reduction is called "condensing", "editing", "prototype selection", etc.

## Results
### DNN Classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/shinome551/condensedMNIST/blob/main/notebook/dnn.ipynb)
The test accuracy for each dataset when training DNN with 4 fully connected layers and ReLU. 

The number of epochs is 100 but the number of iterations is kept the same for each method.

| Sampling Method | Number of Samples |  Test Acc[%] |
|-----------------|:-----------------:|:------------:|
| All data        |       60000       |     97.8     |
| Condensed       |       20953       |     98.0     |
| Random          |       20953       |     96.7     |

### k-Nearest-Neighbor Classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/shinome551/condensedMNIST/blob/main/notebook/kNN.ipynb)
- k=1, L2 distance

| Sampling Method | Number of Samples |  Test Acc[%] |
|-----------------|:-----------------:|:------------:|
| All data        |       60000       |     96.3     |
| Condensed       |       20953       |     95.7     |
| Random          |       20953       |     95.0     |

Note that it is needed Pytorch for dnn.{py, ipynb} and JAX for kNN.{py, ipynb}. It is recommended to use Colab to run the codes or notebooks.

## Citation
```
@inproceedings{bhattacharya1981application,
  title={Application of proximity graphs to editing nearest neighbor decision rule},
  author={Bhattacharya, Binay K and Poulsen, Ronald S and Toussaint, Godfried T}
}
```
