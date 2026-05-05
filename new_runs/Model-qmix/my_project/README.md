# Model-aided FedQMIX

## Requirements

- python 3.7.15 or newer
- numpy 1.21.5 or newer
- torch 1.12.1 or newer
- matplotlib 3.5.3 or newer
- pyswarms 1.3.0 or newer


## Quick Start
### Train
```shell
$ python main.py --map --federated --model --alg --n_agents --tag --total_episodes --device

--map: the map name, RBM/RDM
--federated: whether to use federated learning
--model: whether to use model-aided learning
--alg: the algorithm name, qmix/iql
--n_agents: the number of agents, which is equal to the number of workers in federated learning
--tag: the tag of the experiment
--total_episodes: the total number of collected episodes
--device: the device to run the experiment, cpu/cuda
```
For example, if you want to train a model-aided FedQMIX on the RDM map with 3 agents, you can run the following command:
```shell    
$ python main.py --map=RDM --federated=True --model=True --alg=qmix --n_agents=3 --tag=model_aided_fedqmix --total_episodes=30000
```
### Evaluate
If you want to evaluate the above trained model, you can run the following command:
```shell
$ python evaluate.py --map=RDM --model=True --alg=qmix --tag=model_aided_fedqmix --n_agents=3 
```
### Plot
Once the model training is complete, you can visualize the results using the `Plot.py` script.

### Hints
We also provide an algorithm to improve the model learning performance by using k-means algorithm, you can try it by setting `--model=True --sample_method=kmeans`.

