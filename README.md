# condensedMNIST

## Method
Relative Neighborhood Graphs (RNGs) is one of Proximity Graphs. RNGs represent adjacency between nodes as their edges. 
When creating a RNG, take the following steps:


## Results
### DNN Classification(dnn.py)
The test accuracy when training DNN has 4 fully connected layers on each dataset. The number of epochs is 40 and the number of iterations on each training is made the same.

| Mode          | Number of Samples |   Test Acc   |
|---------------|:-----------------:|:------------:|
| Full          |       60000       |     92.2     |
| Condensed     |       20953       |     92.0     |
| Random        |       20953       |     91.1     |

### k-Nearest-Neighber Classification
coming soon...