import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MessageLength(BaseEstimator, TransformerMixin):
    '''
    Calculates message length, which is later used 
    in the machiene learning pipeline
    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(pd.Series(X).apply(lambda x: len(x)).values)
        # return X.apply(lambda x: len(x)) didn't work. 
        # This is because the Custom Transformer returns only one row instead of 19584.
        # The custom transformer is in parallel to the tfidftransformer, So the number 
        # of observations from both the transformer should be same.