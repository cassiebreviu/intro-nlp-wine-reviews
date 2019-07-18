#%%
## import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump, load

#### Create Functions ####
def loadData():
    csv_file = r"C:\Code\Demos\intro-nlp-wine-reviews\dataset\winemag-review.csv"

    # Read csv file into dataframe
    df = pd.read_csv(csv_file)
    return df

def getQuality(points):
    if(points <= 85):
        return 'bad'
    elif(points<=90 ):
        return 'ok'
    elif(points<=95):
        return 'good'
    elif(points<=100):
        return 'great'
    else:
        return 'If this gets hit, we did something wrong!'



def getPriceRange(price):
    if(price <= 30):
        return '1-30'
    elif(price<=50):
        return '31-50'
    elif(price<=100): 
        return '51-100'
    elif(math.isnan(price)):
        return '0'
    else:
        return 'Above 100'


def get_vector_feature_matrix(description):
    #vectorizer = CountVectorizer(lowercase=True, stop_words="english", max_features=5)
    vectorizer = CountVectorizer(lowercase=True, stop_words="english", max_features=5000)
    #vectorizer = CountVectorizer(lowercase=True, stop_words="english",ngram_range=(1, 2), max_features=5000)
    #vectorizer = CountVectorizer(lowercase=True, stop_words="english", tokenizer=stemming_tokenizer) 
    vector = vectorizer.fit_transform(np.array(description))
    return vector, vectorizer

#### Model Testing Helper ####

def testModel(vectorizer, model, testDescription):
    x = vectorizer.transform(np.array([testDescription]))
    proba = model.predict_proba(x)
    classes = model.classes_
    result = pd.DataFrame(data=proba, columns=classes)
    topPrediction = result.T.sort_values(by=[0], ascending = [False])
    return topPrediction

def getPrediction(description, modelName, vectName):
    
    trained_lr = loadFile(modelName)
    vectorizer = loadFile(vectName)
    x = vectorizer.transform(np.array([description]))
    #result = trained_lr.predict(x)
 
    proba = trained_lr.predict_proba(x)
    classes = trained_lr.classes_
    result = pd.DataFrame(data=proba, columns=classes)
    topPrediction = result.T.sort_values(by=[0], ascending = [False])
    return(topPrediction.to_json(orient='index'))

#### Model Persistance Helpers ####
def exportFile(fileName, file):
    with open('models/'+fileName+'.pkl', 'wb') as fid:
        dump(file, fid)
    print (fileName + " file exported")

def loadFile(fileName, file):
    with open(fileName, 'rb') as file_model:
        return load(file)

print('Functions created')

#%% #Dont run this cell, its just here to break it off from the above cell
def __init__():
#%% Load data and add calculated columns
    df = loadData()
    df['quality'] = df['points'].apply(getQuality)
    df['priceRange'] = df['price'].apply(getPriceRange)
    df.head()
#%% Create vectorizer, features and label
    vector, vectorizer = get_vector_feature_matrix(df['description'])
    features = vector.todense()
    label = df['quality'] 
    #label = df['priceRange']
    #label = df['variety']
    print('Features and labels created')
#%% Split Data
    X, y = features, label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('Data Split')
#%% Train Model
    lr = LogisticRegression(multi_class='ovr',solver='lbfgs')
    model = lr.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print ("Model trained with a Accuracy of {}".format(accuracy))
#%% Export model to file
    exportFile('qualityModel',model)
    exportFile('qualityVect', vectorizer)
