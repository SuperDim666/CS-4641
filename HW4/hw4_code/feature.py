import numpy as np


def create_nl_feature(X):
    '''
    TODO - Create additional features and add it to the dataset
    
    returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    new_features = np.zeros((X.shape[0],1))
    for i in range(X.shape[0]):
        new_features[i] = X[i][0] * X[i][1]
    return np.append(X, new_features, axis = 1)
    #raise NotImplementedError


    
