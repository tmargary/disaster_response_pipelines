import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

sys.path.append("../models")
from message_length import MessageLength

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    df = df.iloc[:50,:]
    X = df['message']
    Y = df.iloc[:,4:].astype('int64')
    category_names = list(np.unique(Y))
    
    return X, Y, category_names


def tokenize(text):
    # tokenization function to process the text data
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return tokens

# class MessageLength(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return pd.DataFrame(pd.Series(X).apply(lambda x: len(x)).values)
#         # return X.apply(lambda x: len(x)) didn't work. 
#         # This is because the Custom Transformer returns only one row instead of 19584.
#         # The custom transformer is in parallel to the tfidftransformer, So the number 
#         # of observations from both the transformer should be same.
        
def build_model():
    # building a machine learning pipeline with GridSearchCV parameters
    parameters = {
        'clf__estimator__criterion': ('gini', 'entropy'),
        'clf__estimator__bootstrap': (True, False)
    }
        
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('txt_len', MessageLength())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3) #added verbose=3
    
    return cv


def evaluate_model(cv, X_test, Y_test, category_names):
    # evaluating the model based on the overall accuracy
    Y_pred = cv.predict(X_test)
    accuracy = (Y_pred == Y_test).mean().mean()
    return accuracy


def save_model(model, model_filepath):
    # exporting the model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))
    
    #with open(model_filepath, 'wb') as file:
    #    pickle.dump(model, file)
        
        
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
       

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()