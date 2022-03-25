import pandas as pd
import numpy as np
import os
import pickle

import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from feature_extraction import Feature_Extractor

class SVM_Classifier():
    
    def __init__(self, features = np.array([]), labels = [], param_grid = {},  orb_des = None, model_path = None, n_components = 500):
        
        self.model_path = model_path

        if self.model_path:
            self.svm_model = self.load_model(self.model_path)
        
        else:
            self.features = features
            self.labels = labels
            self.n_components = n_components
            self.param_grid = param_grid
            self.orb_des = orb_des
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.features, self.labels)
            self.svm_model = self.model_fit(self.param_grid, self.X_train, self.y_train)
            self.save_model(self.svm_model)
            self.evaluate_model(self.svm_model, self.X_test, self.y_test)
        
        # self.matches = self.match_orb_features(self.orb_des)
        
    
    
    def split_data(self, scaled_features, labels):
        
        X = pd.DataFrame(scaled_features)
        y = pd.Series(labels)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=.3,
                                                            random_state=1234123,
                                                            shuffle = True)

        # look at the distrubution of labels in the train set
        print(pd.Series(y_test).value_counts())
        
        return X_train, X_test, y_train, y_test
    
    def model_fit(self, param_grid, X_train, y_train):
        
        # define support vector classifier
        svm = SVC(probability=True, verbose= True)
        model = GridSearchCV(svm, param_grid)
        
        # fit model
        model.fit(X_train, y_train)

        return model
    
    def evaluate_model(self, model, X_test, y_test):
        
        y_pred = model.predict(X_test)

        # calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print('Model accuracy is: ', accuracy)

        return y_pred
    
    
    def save_model(self, model):
        
        with open('./models/model1.pkl', 'wb') as f:
            pickle.dump(model, f)
                    
    def load_model(self, path):
           
        with open(path, 'rb') as f:
           model = pickle.load(f)
        
        return model
        
    def match_orb_features(self, des1):
        """Match Orb Features using the Bf matcher

        Args:
            des1 (numpy array): descriptors of range image at hand

        Returns:
            matches_num [dict]: [keys are the filenames in the feature database and value is the number of matches]
        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
        
        directory_str = "./orb_desc/"

        directory = os.fsencode(directory_str)

        matches_num = {}

        for file in os.listdir(directory):
            
            filename = os.fsdecode(file)

            if filename != ".DS_Store":
                
                des2 = np.loadtxt(directory_str + filename, dtype=np.uint8)
                matches = bf.match(des1, des2)

                matches = sorted(matches, key = lambda x:x.distance)

                matches_num[filename] = len(matches) / len(des1)
            
        return matches_num
    
    def plot_matches(self):
        
        sorted_matches = dict((sorted(self.matches.items(), key=lambda item: item[1]))) 
        print(sorted_matches)


if __name__ == "__main__":
    
    path = "./range_images/"
    
    param_grid={'C':[0.1,1],'gamma':[0.0001,0.001,0.1],'kernel':['rbf','poly']}

    feature_extractor = Feature_Extractor(data_path = path, orb = False) 

    features, labels = feature_extractor.features, feature_extractor.labels

    classifier = SVM_Classifier(features, labels, param_grid)