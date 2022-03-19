# HGNN

This repository is the implementation of our paper's model 'Modeling Factorial User Preference with Hierarchical Graph Neural Network for Enhanced Sequential Recommendation'.

### Files in the folder

- `data/`
  - `steam/`
    - `steam_reviews.json.gz`: raw rating file and metadata of Steam dataset;
    - `preprocess.py`: the data preprocess script;
  - `movielens/`
    - `ratings.dat`: raw rating file of Movielens dataset(UserID::MovieID::Rating::Timestamp);
    - `movies.dat`: genre file of Movielens dataset(MovieID::Title::Genres);
    - `preprocess.py`: the data preprocess script;
- `src/`: implementation of HGNN.


### Required packages
The code has been tested running under Python 3.6.7, with the following packages installed (along with their dependencies):
- torch == 1.7.1+cu101
- numpy == 1.18.5

### Running Procedure

#### Prepare data
Steam and Movielens dataset can be respectively downloaded from 'http://cseweb.ucsd.edu/jmcauley/datasets.html#steam_data/' and 'https://grouplens.org/datasets/movielens/'. 
Please first put data files of Steam and Movielens into `data/steam` and `data/movielens`, then run `preprocess.py` to process data preparation.

#### Run HGNN
```
$ cd src
$ python main.py
```
The settings of datasets and parameters can be altered in `config.py`. 