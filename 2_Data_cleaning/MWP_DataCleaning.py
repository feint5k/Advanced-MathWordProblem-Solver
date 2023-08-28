import pandas as pd
import numpy as np
import re
import json
import nltk
from word2number import w2n
from nltk.stem.snowball import SnowballStemmer

df = pd.read_csv('./Cleaned Data/trainData_univariable.csv')

df = df[np.invert(np.array(df['text'].isna()))]
numMap = {"twice": 2, "double": 2, "thrice": 3, "half": "1/2", "tenth": "1/10", "quarter": "1/4", "fifth": "1/5"}
fraction = {"third": "/3", "half": "/2", "fourth": "/4", "sixth": "/6", "fifth": "/5", "seventh": "/7", "eighth": "/8",
            "ninth": "/9", "tenth": "/10", "eleventh": "/11", "twelfth": "/12", "thirteenth": "/13",
            "fourteenth": "/14", "fifteenth": "/15", "sixteenth": "/16", "seventeenth": "/17", "eighteenth": "/18",
            "nineteenth": "/19", "twentieth": "/20", "%": "/100"}

stemmer = SnowballStemmer(language='english')


# Convert fractions to floating point numbers
def convert_to_fl