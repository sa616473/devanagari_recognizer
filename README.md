# Devanagari Recognizer
Objective: To accuratley classify the Devanagari Characters

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


## Table of contents
- Define Problem
    - Can we accuratley classify the Devanagari Characters?
- Discover Data
    - Data Visualization
- Develop solutions
    - Machine Learning Models
    - Hyperparameter tuning
- Deploy solution
    - Flask and Heroku
    
## Defining the problem

#### What is Devanagari language?
- The Nāgarī or Devanāgarī alphabet developed from eastern variants of the Gupta script called Nāgarī, which first emerged during the 8th century. This script was starting to resemble the modern Devanāgarī alphabet by the 10th century, and started to replace Siddham from about 1200

- The name Devanāgarī is made up of two Sanskrit words: deva, which means god, brahman or celestial, and nāgarī, which means city. The name is variously translated as "script of the city", "heavenly/sacred script of the city" or "city of the Gods or priests" 

#### What are we trying to classify
- we are trying to classify the 46 different characters in the language.

 'ka', 'kha', 'ga', 'gha', 'kna', 'cha', 'chha', 'ja', 'jha', 'yna',
       'taamatar', 'thaa', 'daa', 'dhaa', 'adna', 'tabala', 'tha', 'da',
       'dha', 'na', 'pa', 'pha', 'ba', 'bha', 'ma', 'yaw', 'ra', 'la',
       'waw', 'motosaw', 'petchiryakha', 'patalosaw', 'ha', 'chhya',
       'tra', 'gya', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 
       
       
![Image of characters](/reports/figures/devanagari_cons.gif)

![Image of numbers](/reports/figures/index.png)

## Discovering the Data

#### Data Visualization
#### --**Images before pre-processing** --
![Image before preprocessing](/reports/figures/char_plots.png)


#### -- **Images after pre-processing** --
![Image after preprocessing](/reports/figures/char_plots_binary.png)

##### We can see that the images are more clear after pre processing

### Developing Solutions

**NOTE: These models were tested on unseen data.**

-- **1. Densly connected Neural Network(DNN)** --

- We used a simple model at first to see how the model will perform this a simple DNN with one hidden layer.
- Results 


|#|loss |accuracy |layer|
|--|-----|---------|-----|
 |0 |3.012618 |0.853333 |1024.0|
 |1 |3.008728 |0.857029 |512.0|
 |2 |3.196688 |0.668768 |128.0|
 
 - Grpahs of the loss function and accuracy 
 - Layer with 1024 neurons
 ![0](/reports/performance/DNN_1024_layer.png)
 
 - Layer with 512 neurons
 ![1](/reports/performance/DNN_512_layer.png)
 
 - Layer with 128 neurons
 ![2](/reports/performance/DNN_128_layer.png)
 
 -- **2. Convolution Neural Netowork(CNN)** --
 - We used a little complicated model after that.
 - This is a CNN with 5 hidden layers.
 - Result
 
| # |loss |accuracy |layer|
|--|-----|---------|------|
 |0 |0.089015 |0.974710 |128.0|
 |1 |0.221300 |0.940870 |64.0|
 |2 |1.114612 |0.742971 |32.0|
 
 - Graphs of the loss function and accuracy
 - Layer with 128, 64, 32 filters
  ![0](/reports/performance/CNN_128_filter.png)
  
 - Layer with 64, 32, 16 filters
  ![0](/reports/performance/CNN_64_filter.png)

  - Layer with 32, 16, 8 filters
  ![0](/reports/performance/CNN_32_filter.png)
 
 #### OUR BEST MODEL IS CNN WITH 128,64,32 filters.
 
 ### Deploying Solution
 
 - We deployed our solution using Flask and Heroku
 - Here is the link ![https://devanagari-recognizer.herokuapp.com/]

 
 




