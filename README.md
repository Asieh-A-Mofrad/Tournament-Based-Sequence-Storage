# Sequence Storage in Tournament-Based Structures

This code simulates the proposed sequence storage models and retrieval algorithms that introduced in the paper below that is submitted to the [Neural Computation](https://www.mitpressjournals.org/loi/neco):

**Mofrad, A. A., Mofrad, S. A., Yazidi, A., & Parker, M. G.  (2021). On Neural Associative Memory Structures: Storage and Retrieval of Sequences in a Chain of  Tournaments .**

The structures are based on the original work by [Jiang et. al (2016)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7114322) that deals with the problem of storing then retrieving sequences in sparse binary neural networks.

## Getting Started

### Prerequisites

This code has been tested with Python 3.6.5 (Anaconda). You can download it or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([notebook.ipynb - Colaboratory (google.com)](https://colab.research.google.com/github/Asieh-A-Mofrad/Tournament-Based-Sequence-Storage/blob/main/notebook.ipynb) and access to the live view of the notebook. 

:warning: The program might crash if the initial values are not valid

## Process summary

Before running the `main.py`, you can change the initial values in `initialization.py`
By initialization and running `main.py`:

- `result_no`  is used to assign a set of parameters initial values in `initialization.py`  and is used for saving the output

- `structure_ID` is a number between 1-5 that determines the memory and retrieval setting  

- The results will be saved in a pickle file in the "results" folder.

- Results of a previously simulated data can be accessed by its filename. 

- `Learning_set_flag` must be set `True`  in the case that there is no Learning set for the chosen initial values, or one want to generate a new learning set. Otherwise, it is `False`. 

  To see the results for a previous simulation, change `file_name= None` to a file name in the results folder, say `file_name = 'results/Tournament_Winner_2_100.p'`

```python
result_no = 2
structure_ID = 4
file_name= None # previous run, say 'results/Tournament_Winner_2_100.p' 
Learning_set_flag = True # or False
```

## Configuration of Initial values (In initialization.py) 



Each `result_no` can represent a dictionary of a set of parameters that are used for generating learning set and simulate the storing and retrieval processes. 

```python
Initial_values = {
    1: { 
        'memory_type' : memory_type[structure_ID], 
        'c' : 20,
        'k' : 8,
        'L' : 100,
        'r' : 12,  
        'r_fdbk' : 6, 
        'r_explore' : 7,
        'Num_Seq': [10, 300, 600, 900, 1200,  1500,  1800, ... , 14700, 15000],
        'Iter' : 100
          },
    2: {...
    ...}
    ...
    }
```



- The first item in `initial_values` is `memory_type` .`structure_ID`in the `main.py`  must be selected based on the below dictionary.

```python
memory_type = {1 : 'Tournament_Winner', 
               2 : 'Tournament_Cache_Winner',
               3 : 'Tournament_Explore_Winner',
               4 : 'Tournament_Feedback_Winner',
               5 : 'Tournament_Backward_Winner'
        }
```

where

1. **Tournament_Winner**:
   Is a replication of original Tournament-based network that is proposed in  [Jiang et. al (2016)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7114322)

2. **Tournament_Cache_Winner**:

   The learning is similar to the `Tournament_Winner`, the retrieval algorithm uses a temporarily Cache memory to remove a large part of the ambiguities that can occur due to random selections in the previous component retrieval.

3. **Tournament_Explore_Winner**:

   The learning is similar to the `Tournament_Winner`, the retrieval algorithm anticipates what might happen in future steps before making a decision at current step and therefore make a more accurate decision.

4. **Tournament_Feedback_Winner**:

   In this architecture, backward connections in addition to forward connections is used. The retrieval algorithm is updated for this new Tournaments. This can be seen as a generalization of  `Tournament_Winner`.  

5. **Tournament_Backward_Winner**

   The learning is similar to the `Tournament_Feedback_Winner` but the retrieval is from backward; i.e. the given sub-sequence belongs to the end of a sequence.

Other parameters in `initial_values` are as follows:

- `c`: is the number of clusters in the model
- `k`: is the required bits to represent a sub-sequence (component). The number of (nodes) neurons in a cluster equals l=2<sup>k</sup>.
- `L`: the length of sequence (number of components)
- `r`: forward output edges from each active neuron.
- `r_fdbk`: the size of backward edges from each active neuron. Applicable in `Tournament_Feedback_Winner` and `Tournament_Backward_Winner`. 
- `r_explore`: The number of clusters which is used for exploration.  Applicable in `Tournament_Explore_Winner`. `r_explore` must be less than `r-1`.
- `Num_Seq`: is a list of integers as the size of learning set in simulations.
- `Iter`: The number of repetition of retrieval (the error is an average over errors in retrieving `Iter` randomly selected sequences from the learning set)

## License

The code is written by Asieh & Samaneh Abolpour Mofrad under

MIT license (MIT-LICENSE or http://opensource.org/licenses/MIT)

