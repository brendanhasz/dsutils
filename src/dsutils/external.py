"""Classes which depend on external packages

* :class:`.SurpriseRegressor`

"""

import os
import subprocess

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin

import surprise

from nltk import tokenize
from bert_serving.client import BertClient
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA



class SurpriseRegressor(BaseEstimator, RegressorMixin):
    """Uses a suprise model to predict user scores of items
    
    A wrapper sklearn interface for suprise models 
    (https://github.com/NicolasHug/Surprise).
    
    Parameters
    ----------
    model : surprise model
        The surprise model to use (e.g. BaselineOnly, KNNBaseline, or SVD)
    user_col : int or str
        Column of X to use as user IDs.
        Default is 0.
    item_col : int or str
        Column of X to use as item IDs.
        Default is 1.
    score_range : tuple of int or float
        Range of the scores in y.  First element should be lower 
        bound and second element should be upper bound.
        Default is to use the minimum and maximum y values.
    **kwargs
        Additional kwargs are passed to the surprise model constructor.
        
    Examples
    --------
    
    To create a sklearn regressor which uses the surprise's
    implementation of SVD:
    
        model = SurpriseRegressor(model=surprise.SVD,
                                  user_col='user_id',
                                  item_col='item_id')
                                  
    Pass additional kwargs to pass to the suprise model.
    For example, to set the number of factors and the
    number of epochs for training the SVD model,
    
        model = SurpriseRegressor(model=surprise.SVD,
                                  user_col='user_id',
                                  item_col='item_id',
                                  n_factors=50,
                                  n_epochs=1000)
                                  
    Or to set the number of neighbors for the KNN model:
    
        model = SurpriseRegressor(model=surprise.KNNBaseline,
                                  user_col='user_id',
                                  item_col='item_id',
                                  k=50)
                                  
    """
    
    def __init__(self, model=surprise.SVD, user_col=0, item_col=1, score_range=(None, None), **kwargs):
        """Create the regressor"""
        
        # Check inputs
        if not isinstance(user_col, (int, str)):
            raise TypeError('user_col must be a str or an int')
        if not isinstance(item_col, (int, str)):
            raise TypeError('item_col must be a str or an int')
        if not isinstance(score_range, tuple):
            raise TypeError('score_range must be a tuple')
        if len(score_range) != 2:
            raise ValueError('score_range must be a tuple with 2 elements')
        if not all(isinstance(e, (int, float)) for e in score_range):
            raise TypeError('score_range must contain ints or floats')
        if score_range[0] > score_range[1]:
            raise ValueError('First element of score_range must be less than second element')
            
        # Store parameters
        self._user_col = user_col
        self._item_col = item_col
        self._score_range = score_range
        self._kwargs = kwargs
        self._surprise_model = model
        self._model = None


    def fit(self, X, y):
        """Fit the surprise model to data.
        
        Parameters
        ----------
        X : pandas DataFrame
            Table containing user IDs and item IDs.
        y : pandas Series
            Scores.
            
        Returns
        -------
        self
            The fit estimator.
        """

        # Create a pandas DataFrame in suprise format
        X_train = X[[self._user_col, self._item_col]]
        X_train['score'] = y
        
        # Compute the score range
        if self._score_range[0] is None:
            self._score_range[0] = y.min()
        if self._score_range[1] is None:
            self._score_range[1] = y.max()
        
        # Create a suprise Dataset from the dataframe
        reader = surprise.Reader(rating_scale=self._score_range)
        dataset = (surprise.dataset.Dataset
                   .load_from_df(X_train, reader))
        dataset = dataset.build_full_trainset()

        # Fit the model
        self._model = self._surprise_model(**self._kwargs)
        self._model.fit(dataset)
        return self
    

    def predict(self, X, y=None):
        """Predict the scores using the surprise model.
        
        Parameters
        ----------
        X : pandas DataFrame
            Table containing user IDs and item IDs.
            
        Returns
        -------
        y_pred : pandas Series
            Predicted scores.        
        """

        # Check if model has been fit
        if self._model is None:
            raise RuntimeError('model has not been fit')
        
        # Create a pandas DataFrame in suprise format
        X_test = X[[self._user_col, self._item_col]]
        X_test['score'] = np.nan
        
        # Create a suprise Testset from the dataframe
        reader = surprise.Reader(rating_scale=self._score_range)
        testset = (surprise.dataset.Dataset
                   .load_from_df(X_test, reader))
        testset = testset.build_full_trainset().build_testset()
        
        # Use suprise model to predict scores
        preds = self._model.test(testset) #returns a list of "Prediction" objs...
        preds = [pred[3] for pred in preds]
        return pd.Series(data=np.array(preds),
                         index=X.index)
        
        
    def fit_predict(self, X, y):
        """Fit the surprise model and predict the scores.
        
        Parameters
        ----------
        X : pandas DataFrame
            Table containing user IDs and item IDs.
        y : pandas Series
            True scores.
            
        Returns
        -------
        y_pred : pandas Series
            Predicted scores.
        """
        return self.fit(X, y).predict(X)



class BertEncoder(BaseEstimator, TransformerMixin):
    """Use a pre-trained BERT model to embed sentences and paragraphs.

    Embeds columns of text data using a pre-trained BERT model.  Adds columns
    corresponding to the embedding values (reduced in dimensionality using
    PCA), and optionally removes the original columns containing the text.
    Embeds paragraphs by taking the average embedding of each sentence in the
    paragraph.

    Parameters
    ----------
    cols : str or list of str
        Column(s) to encode.
    clean_text_fn : str
        Function to use to clean the text data. 
        Default is to simply replace commas with space, and 
        convert to lowercase.
    n_pc : int
        Number of top principal components to keep.
        Default = 5
    delete_old : bool
        Whether to delete the text columns whose values
        were encoded.  Default = True
    bert_model : str
        What BERT model to use.
        See https://bert-as-service.readthedocs.io/en/latest/section/get-start.html#download-a-pre-trained-bert-model
        Default = uncased_L-12_H-768_A-12
    bert_url : str
        Url from which to fetch the BERT model.
        See https://bert-as-service.readthedocs.io/en/latest/section/get-start.html#download-a-pre-trained-bert-model
        Default = http://storage.googleapis.com/bert_models/2018_10_18/
    """
    
    def _clean_text(text):
        if isinstance(text, float):
            return ' '
        return text.replace(',', ' ').lower()


    def __init__(self, 
                 cols, 
                 clean_text_fn=_clean_text,
                 n_pc=5,
                 delete_old=True,
                 bert_model='uncased_L-12_H-768_A-12',
                 bert_url='http://storage.googleapis.com/bert_models/2018_10_18/',
                 model_dir=None):

        # Check types
        if not isinstance(cols, (list, str)):
            raise TypeError('cols must be a list or a string')
        if isinstance(cols, list):
            if not all(isinstance(c, str) for c in cols):
                raise TypeError('each element of cols must be a string')
        if not callable(clean_text_fn):
            raise TypeError('clean_text_fn must be a callable')
        if not isinstance(n_pc, int):
            raise TypeError('n_pc must be an int')
        if n_pc<1:
            raise ValueError('n_pc must be 1 or greater')
        if not isinstance(delete_old, bool):
            raise TypeError('delete_old must be a bool')
        if not isinstance(bert_model, str):
            raise TypeError('bert_model must be a str')
        if not isinstance(bert_url, str):
            raise TypeError('bert_url must be a str')
        if model_dir is not None and not isinstance(model_dir, str):
            raise TypeError('model_dir must be a str or None')
            
        # Store parameters
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.clean_text_fn = clean_text_fn
        self.n_pc = n_pc
        self.pca = dict()
        self.delete_old = delete_old
        self.bert_model = bert_model
        self.bert_url = bert_url
        self.model_dir = model_dir

        # Directory in which to store BERT model
        if model_dir is None:
            process = subprocess.Popen('pwd', stdout=subprocess.PIPE)
            output, error = process.communicate()
            model_dir = output[:-1].decode('utf-8')+'/'
        else:
            if model_dir[-1]!='/':
                model_dir = model_dir+'/'
        
        # Get the BERT model and start the server if not already downloaded
        if not os.path.isfile(model_dir+bert_model+'.zip'):

            # Download the BERT model
            cmd = 'wget '+bert_url+bert_model+'.zip -P '+model_dir
            subprocess.check_call(cmd.split())
            
            # Unzip the model
            cmd = 'unzip '+model_dir+bert_model+'.zip'
            subprocess.check_call(cmd.split())
            
            # Start the BERT server
            cmd = 'bert-serving-start -model_dir '+model_dir+bert_model
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE) 
            
        # Start the BERT client
        self.bc = BertClient(check_length=False)
        
        
    def _bert_embed_paragraphs(self, paragraphs):
        """Embed paragraphs by taking the average embedding of each sentence.
        
        Converts paragraphs (list of lists of strings), to a single list
        of strings, and embeds each sentence with BERT.  Then, takes the
        average sentence embedding for sentences in each paragraph.

        Parameters
        ----------
        paragraphs : list of lists of str
            The paragraphs.  Each element should correspond to a paragraph
            and each paragraph should be a list of str, where each str is 
            a sentence.

        Returns
        -------
        embeddings : numpy ndarray of size (len(paragraphs), 768)
            The paragraph embeddings
        """

        # Covert to single list
        sentences = []
        ids = []
        for i in range(len(paragraphs)):
            sentences += paragraphs[i]
            ids += [i]*len(paragraphs[i])

        # Embed the sentences
        embeddings = self.bc.encode(sentences)

        # Average by paragraph id
        Np = len(paragraphs) #number of paragraphs
        n_dims = embeddings.shape[1]
        embeddings_out = np.full([Np, n_dims], np.nan)
        ids = np.array(ids)
        the_range = np.arange(len(ids))
        for i in range(n_dims):
            temp = coo_matrix((embeddings[:,i], (ids, the_range))).mean(axis=1)
            embeddings_out[:temp.shape[0], i] = temp[:, 0].ravel()
        return embeddings_out

        
    def _bert_encode(self, series, name, training):
        """Encode a single column of text data with BERT"""

        # Clean the text
        data = [self.clean_text_fn(e) for e in series.tolist()]
        
        # Convert paragraphs to lists of sentences
        data = [tokenize.sent_tokenize(s) for s in data]
        
        # Embed the paragraphs
        embeddings = self._bert_embed_paragraphs(data)
        
        # Compute PCA on training data
        if training:
            self.pca[name] = PCA(n_components=self.n_pc)
            self.pca[name] = self.pca[name].fit(embeddings)
        
        # Apply PCA to reduce dimensionality
        embeddings = self.pca[name].transform(embeddings)
        
        # Return the embeddings
        return embeddings
        

    def fit(self, X, y):
        """Nothing needs to be done here"""
        return self

    
    def transform(self, X, y=None):
        """Perform the BERT encoding.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        
        # Create output dataframe
        Xo = X.copy()

        # Encode each column
        for col in self.cols:
            embeddings = self._bert_encode(X[col], col, y is not None)
            for iC in range(self.n_pc):
                Xo[col+'_'+str(iC)] = embeddings[:, iC]
            if self.delete_old:
                del Xo[col]

        # Return encoded DataFrame
        return Xo
      
            
    def fit_transform(self, X, y=None):
        """Perform the BERT encoding.
        
        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_columns)
            Independent variable matrix with columns to encode
        y : pandas Series of shape (n_samples,)
            Dependent variable values.

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)