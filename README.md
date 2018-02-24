# Active Neural Localization
This is a PyTorch implementation of the ICLR-18 paper:

[Active Neural Localization](https://arxiv.org/abs/1801.08214)<br />
Devendra Singh Chaplot, Emilio Parisotto, Ruslan Salakhutdinov<br />
Carnegie Mellon University

Project Website: https://devendrachaplot.github.io/projects/Neural-Localization

### This repository contains:
- Code for the Maze2D Environment which generates random 2D mazes for active localization.
- Code for training an Active Neural Localization agent in Maze2D Environment using A3C.

## Dependencies
- [PyTorch](http://pytorch.org) (v0.3)

## Usage

### Training
For training an Active Neural Localization A3C agent with 16 threads on 7x7 mazes with maximum episode length 30:
```
python a3c_main.py --num-processes 16 --map-size 7 --max-episode-length 30 --dump-location ./saved/ --test-data ./test_data/m7_n1000.npy
```
The code will save the best model at `./saved/model_best` and the training log at `./saved/train.log`. The code uses `./test_data/m7_n1000.npy` as the test data and makes sure that any maze in the test data is not used while training.

### Evaluation
After training, the model can be evaluated using:
```
python a3c_main.py --num-processes 0 --evaluate 1 --map-size 7 --max-episode-length 30 --load ./saved/model_best --test-data ./test_data/m7_n1000.npy
```

### Pre-trained models
The `pretrained_models` directory contains pre-trained models for map-size 7 (max-episode-length 15 and 30), map-size 15 (max-episode-length 20 and 40) and map-size 21 (max-episode-length 30 and 60). The test data used for training these models is provided in the `test_data` directory.

For evaluating a pre-trained model on maze size 15x15 with maximum episode length 40:
```
python a3c_main.py --num-processes 0 --evaluate 1 --map-size 15 --max-episode-length 40 --load ./pretrained_models/m15_l40 --test-data ./test_data/m15_n1000.npy
```

### Generating test data
The repository contains test data of map-sizes 7, 15 and 21 with 1000 mazes each in the `test_data` directory. 

For generating more test data:
```
python generate_test_data.py --map-size 7 --num-mazes 100 --test-data-location ./test_data/ --test-data-filename my_new_test_data.npy
```
This will generate a test data file at `test_data/my_new_test_data.npy` containing 100 7x7 mazes.

### All arguments
All arguments for a3c_main.py:
```
  -h, --help            show this help message and exit
  -l L, --max-episode-length L
                        maximum length of an episode (default: 30)
  -m MAP_SIZE, --map-size MAP_SIZE
                        m: Size of the maze m x m (default: 7), must be an odd
                        natural number
  --lr LR               learning rate (default: 0.001)
  --num-iters NS        number of training iterations per training thread
                        (default: 10000000)
  --gamma G             discount factor for rewards (default: 0.99)
  --tau T               parameter for GAE (default: 1.00)
  --seed S              random seed (default: 1)
  -n N, --num-processes N
                        how many training processes to use (default: 8)
  --num-steps NS        number of forward steps in A3C (default: 20)
  --hist-size HIST_SIZE
                        action history size (default: 5)
  --load LOAD           model path to load, 0 to not reload (default: 0)
  -e EVALUATE, --evaluate EVALUATE
                        0:Train, 1:Evaluate on test data (default: 0)
  -d DUMP_LOCATION, --dump-location DUMP_LOCATION
                        path to dump models and log (default: ./saved/)
  -td TEST_DATA, --test-data TEST_DATA
                        Test data filepath (default: ./test_data/m7_n1000.npy)
```



## Cite as
>Chaplot, Devendra Singh, Parisotto, Emilio and Salakhutdinov, Ruslan.
Active Neural Localization. 
In *International Conference on Learning Representations*, 2018. 
([PDF](http://arxiv.org/abs/1801.08214))

### Bibtex:
```
@inproceedings{chaplot2018active,
  title={Active Neural Localization},
  author={Chaplot, Devendra Singh and Parisotto, Emilio and Salakhutdinov, Ruslan},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```

## Acknowledgements
The implementation of A3C is borrowed from https://github.com/ikostrikov/pytorch-a3c.
