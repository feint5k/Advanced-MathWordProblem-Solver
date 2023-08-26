import pandas as pd
import numpy as np
import re
import json
import nltk
from word2number import w2n
from nltk.stem.snowball import SnowballStemmer

df = pd.read_csv('./Cleaned Data/trainData_univariable.csv')

df = df[np.invert(np.array(df['text'].isna()))]
numMap = {"twice": 2, "double": 2, "thrice": 3, "half": "1/2", "tenth": "1/10", "quarter": "1/4", "fifth":