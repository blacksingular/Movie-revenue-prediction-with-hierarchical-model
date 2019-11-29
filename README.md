# SI699: Movie Revenue Prediction: Based on hierarchical regression model

Code for SI 699 course project: Using hierechical regression model to predcit the movie revenue based on data from IMDB public dataset. The final report will be followed later.

## Getting Started

### Installing

1.To setup on your local machine:
Install Anaconda with Python >= 3.6.

2.Clone the repository
```
git clone https://github.com/blacksingular/SI699.git
```

3.some package pre-installed:
```
pip install tqdm
pip install -U scikit-learn 
```

### Update Data Set

we grabed 3259 movies metadata (2008 - 2018) from [IMDB.com](https://www.imdb.com/) using [OMDB API](http://www.omdbapi.com/).
If you would like to grab latest data. Run crawler.py
```
python crawler.py
```



### Running the tests
To train the model as well as do some test. Run:
```
python improvement.py
```

## Results

we evaluated our model on 3259 movies from 2008 to 2019 using Mean Absolute Rrror [(MAE)](https://en.wikipedia.org/wiki/Mean_absolute_error) and Symmetric Mean Absolute Percentage Error 
[(SMAPE)](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)

|               | MAE ( $ M )         | SNAPE  |
| ------------- |:----------------: | ---------------:|
| Train         | $ 7.1 M        | 0.451 |
| Test          | $ 23.5 M       | 0.905 |




### Authors
Team Lucy: Jiazhao Li, Yun Gao -- Dept. EECS of University of Michigan. 
 

### Acknowledgments

Thanks to project mentor: Prof. Qiaozhu Mei for inspiration and suggestions.

