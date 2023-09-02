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
def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


# Convert infix expression to suffix expression
def postfix_equation(equ_list):
    stack = []
    post_equ = []
    op_list = ['+', '-', '*', '/', '^']
    priori = {'^': 3, '*': 2, '/': 2, '+': 1, '-': 1}
    for elem in equ_list:
        if elem == '(':
            stack.append('(')
        elif elem == ')':
            while 1:
                op = stack.pop()
                if op == '(':
                    break
                else:
                    post_equ.append(op)
        elif elem in op_list:
            while 1:
                if not stack:
                    break
                elif stack[-1] == '(':
                    break
                elif priori[elem] > priori[stack[-1]]:
                    break
                else:
                    op = stack.pop()
                    post_equ.append(op)
            stack.append(elem)
        else:
            post_equ.append(elem)
    while stack:
        post_equ.append(stack.pop())
    return post_equ


# Identification and filtering of univaria