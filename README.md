# SDOpt:Semi-Definite Optimization Layer for Index Tracking


This is the implementation of the paper  "SDOpt:Semi-Definite Optimization Layer for Index Tracking". All the implementations are written in Python3.

## Set up environment

To run the code, first you have to set up a conda environment. Once you have [Anaconda installed](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), run the following command:
```
conda env create --name lodlenv --file=dflenv.yml
```
Once the environment has been created, load it using the command
```
conda activate lodlenv
```



## Running Different Domains

We now present the default parameters required to run SDOpt approach.

'''
python main.py --exp_num 0 --loss dfl --market nasdaq100 --cardinality 1.0 --save_mode True 
'''



## Running Different Approaches

 To run the domains above with a specific approach, set the `--loss` parameter to the input corresponding to that approach.

| Method      | Corresponding Input |
| ----------- | ----------- |
| 2-Stage     | mse         |
| DFL         | dfl         |

Similarly, to run different market or cardinality, set the `--market` and '--cardinality' parameter 

Additionally, 'save_mode' parameter can be used for saving cvxpy problems for computational efficiency
