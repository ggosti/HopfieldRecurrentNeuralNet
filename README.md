# HopfieldRecurrentNeuralNet

Implementation of a discrete-time recurrent neural network with binary neurons.

Requirements
------------

The code uses Python2.7 and the following Python open source packages:
- NumPy
- Matplotlib
- NetworkX

You may install the Python requirements with `pip install -r requirements.txt`.

Generate Transitions
------------

This implementation considers a discrete-time recurrent neural network with binary neurons.
The number of neurons `N` can be set, and the neurons can have activation state either {0,1} if `typ = 0`,
or {-1,1} if `typ = 1`. The activation function can only be a step function with a certain threshold thr.

In the following text it is assumed that the following packages are loaded:
```python
import learningRNN as lrnn
import numpy as np
import matplotlib.pyplot as plt
```

It is useful to set the following parameters:
```python
N= 16 # Number of neurons
typ = 0 # typ = 0 neurons with binary activation states {0,1}, typ = 1  neurons with states {-1,1}.
        # typ=1 --> {-1,1}    typ=0 --> {0,1} 
thr = 0 # activation function threshold
```


Generate a random network with arbitrary degree of asymmetry and sparsity as described in 
(Folli et al., [2018](https://doi.org/10.1016/j.neunet.2018.04.003))
:
```python
net = get_connectivity_matrix(5555,N,rho=0.9,epsilon=1.)
```

Given that a binary vector can be cast as an integer there is a 
mapping between each binary vector and a integer. We use this mapping to index all the binary vector.
Generate an initial state from an index.
```python
initial_state = lrnn.stateIndex2stateVec(17,N,typ)
```

We can get back to the index:
```python
initial_state_index = lrnn.stateVec2stateIndex(initial_state, N, typ)
```

Get the next state given the objective network and an `initial_state`
```python
transition = lrnn.transPy(initial_state,net, N, typ, thr)
```

Given a list of initial states `initial_state_list`, get a list with the corresponding transition
```python
initial_state_list = lrnn.stateIndex2stateVecSeq([19,2001,377], N, typ)
transition_list = lrnn.transPy(initial_state_list,net, N, typ, thr)
```


## References

Folli, V., Gosti, G., Leonetti, M., &#38; Ruocco, G. (2018). Effect of dilution in asymmetric recurrent neural networks. <i>Neural Networks</i>, <i>104</i>, 50–59. https://doi.org/10.1016/j.neunet.2018.04.003</div>

