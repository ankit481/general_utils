# import libraries
import _pickle as pickle
from datetime import date, datetime, timedelta
import dask.dataframe as dask_dataframe
from dask.multiprocessing import get
import gspread
import gspread_dataframe
import itertools
import numba
import nltk
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
import pandas as pd
import pymysql

# create generic regular expression compilor; will be used to remove special characters.
#regex = re.compile('[^a-zA-Z0-9 ]')
import re
regex = re.compile('([^\s\w]|_)+')

import sqlalchemy 

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer

from time import time
from tqdm import tqdm
import warnings

# xgboost
from xgboost.sklearn import XGBClassifier

# load credentials
with open('/home/ankit/general_utils/credentials/credentials.p','rb') as fp:
    credentials=pickle.load(fp)
    
def get_connection():
    return pymysql.connect(host=credentials['host'],
                           user=credentials['user'],
                           password=credentials['password'],
                           db=credentials['db'],
                           port=3306,
                           charset='utf8mb4',
                           cursorclass=pymysql.cursors.SSDictCursor)


def get_database_cursor():
    connection = get_connection()
    # create the cursor
    cursor = connection.cursor(cursor=pymysql.cursors.SSDictCursor)
    
    return cursor

def execute_sql_query(sql):
    # get database cursor
    database_cursor = get_database_cursor()
    # execute sql query
    database_cursor.execute(sql)
    # create data frame
    df = pd.DataFrame.from_dict(database_cursor.fetchall())
    # return dataframe
    return df

# custom tokenizer
def tokenize(text):
    
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in sent.split() if len(word) > 1 and 'emoji' not in word ] 
    return tokens

# preprocess text
def preprocess_text(df_column):
    df_column=[regex.sub('', x.lower()) for x in df_column]
    return df_column

# load model
def load_model(model_path):
    with open(model_path,'rb') as fp:
        model=pickle.load(fp) 
    return model

# save model
def save_model(model_path, model):
    with open(model_path,'wb') as fp:
        pickle.dump(model,fp)
    return 'model saved'

def evaluate_model(y_test, prediction):
    print("Accuracy:", accuracy_score(y_test, prediction))
    print("Precision:", precision_score(y_test, prediction))
    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))
    print("\nformat of confusion matrix:")
    print("[true negative, false positive]")
    print("[false negative, true positive]")

    '''
    [[ true_negatives   false_positives]
     [ false_negatives  true_positives]]
    '''

# create dataframe partitions
def create_df_partitions(df, NUM_PARTITIONS):
    return dask_dataframe.from_pandas(df, npartitions=NUM_PARTITIONS)

# authorize google credentials
def authorize_google_credentials():
    SCOPE = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive',
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name('/home/ankit/general_utils/credentials/google_cred.json', SCOPE)
    google_client = gspread.authorize(credentials)
    
    return google_client

# get ml db connection
def get_db_connection():
    username = credentials['user']
    password = credentials['password']
    host = credentials['host']
    port = 3306
    name = 'insert db name'
    connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}:{3}/{4}'.
                                                   format(username, password,host, 
                                                          port, name))
    return connection

# write dataframe to mysql database
def write_to_db(df,TABLE_NAME,IF_EXISTS,CHUNK_SIZE):
    connection=get_db_connection()
    df.to_sql(con=connection,name=TABLE_NAME,if_exists=IF_EXISTS,chunksize=CHUNK_SIZE)
    
