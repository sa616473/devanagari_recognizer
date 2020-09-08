import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

#Getting the data
def get_data(filename ='../data/raw/data.csv' ):
    data = pd.read_csv(filename).values
    #Shuffling the data
    np.random.shuffle(data)
    labels = data[:, -1]
    images = data[:,:-1]
    #Converting the ints to floats for preprocessing the data
    images = images.astype('float').reshape(images.shape[0], 32,32)
    return images, labels

#preprocessing the data
def pre_processing(char_data, char_labels):
    
    char_data = char_data/255
    classes = np.unique(char_labels)
    for i in range(46):
        classes[i] = classes[i].split('_')[-1]
    
    encoder = LabelEncoder()
    char_labels = encoder.fit_transform(char_labels)
    
    return char_data, char_labels, classes
    
    
    