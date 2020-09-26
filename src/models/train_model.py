from tensorflow.keras.models import model_from_json
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten,Conv2D, MaxPool2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import sys
sys.path.insert(1, '../src/')

from visualization import visualize as vs

import pandas as pd
import numpy as np


def save_model(model, title=''):
    '''
    This function saves the model into HDF5 format and weights
    into a JSON file
    '''
    model_json = model.to_json()
    with open("../src/models/saved_models/model_{}.json".format(title), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../src/models/saved_models/model_{}.h5".format(title))
    print("Saved model to disk")

def callbacks():
    '''
    Early stopping callbacks
    '''
    callback = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True)
    return callback

#1st model simple DNN
def dnn_model(training=(), val=(), test=(), epochs = 100, layers=[1024, 512, 128]):
    #Performance dictionary
    performance = {}
    
    #Earlystopping
    callback = callbacks()
    
    i=0
    
    for layer in layers:
        #model
        model = Sequential()
        model.add(Flatten(input_shape=(32,32)))
        model.add(Dense(layer, activation='relu'))
        model.add(Dense(46, activation='softmax'))
        
        #Compliling the model
        model.compile(optimizer='adam', 
                      loss=SparseCategoricalCrossentropy(from_logits=True), 
                      metrics=['accuracy'])
        
        #Training the model
        history = model.fit(training[0], training[1],
                  epochs=epochs,
                  callbacks=[callback],
                 validation_data=(val[0], val[1]), verbose=3)
        
        #Visualizing the model
        vs.training_visualize(history, title='DNN {} layer'.format(layer))
        
        #Save model
        save_model(model, title='{}'.format(layer))
        
        #Evaluate model
        evaluation = model.evaluate(test[0], test[1])
        
        #Record Performance.
        performance[i] = [evaluation[0], evaluation[1], layer]   
        i = i+1
    performance = pd.DataFrame(data=performance)
    performance = performance.transpose()
    performance.columns = ['loss', 'accuracy', 'layer']
    performance.to_csv('../src/models/performance_DNN.csv')
    
    return performance

def cnn_model(train=(), test=(), val=(),epochs = 100, layers=[128,64,32]):
    performance = {}
    
    checkpoint_filepath = '../src/models/checkpoints'
    
    model_checkpoint_callback=ModelCheckpoint(filepath=checkpoint_filepath,save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)
    
    callback = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True)
    
    train_data = train[0]
    train_data = train_data[:,:,:,np.newaxis]
    val_data = val[0]
    val_data = val_data[:,:,:,np.newaxis]
    test_data = test[0]
    test_data = test_data[:,:,:,np.newaxis]
    
    i=0
    for layer in layers:
        model = Sequential()
        #layer 1
        model.add(Conv2D(layer, (3,3), activation='relu', input_shape=(32,32,1)))
        model.add(MaxPool2D())
        #layer 2
        model.add(Conv2D(layer/2, (3,3), activation='relu'))
        model.add(MaxPool2D())
        #layer 3
        model.add(Conv2D(layer/4, (3,3), activation='relu'))
        model.add(MaxPool2D())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(46, activation='softmax'))

        model.compile(loss=SparseCategoricalCrossentropy(), 
                 optimizer='adam',
                 metrics=['accuracy'])
        history = model.fit(train_data, train[1],
                  epochs=epochs,
                  callbacks=[callback, model_checkpoint_callback],
                 validation_data=(val_data, val[1]))
        
        vs.training_visualize(history, title='CNN {} filter'.format(layer))
        
        save_model(model, title='{}_{}'.format(layer, int(layer/2)))
        
        evaluation = model.evaluate(test_data, test[1])
        
        performance[i] = [evaluation[0], evaluation[1], layer]   
        i = i+1
    performance = pd.DataFrame(data=performance)
    performance = performance.transpose()
    performance.columns = ['loss', 'accuracy', 'layer']
    performance.to_csv('../src/models/performance_CNN.csv')
    
    return performance
    