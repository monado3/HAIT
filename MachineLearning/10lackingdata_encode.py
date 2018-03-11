# In[]
import numpy as np
import pandas as pd
from numpy import nan

# In[]
df = pd.DataFrame({'DATE':  [20170801, 20170801, 20170802, 20170802, 20170803, 20170803, 20170803, 20170805, 20170805, 20170805],
                   'NECK':  [41, nan, 38, 46, nan, 37, nan, 38, nan, 42],
                   'BODY':  [84, 92, nan, 90, nan, 64, 78, 74, 82, 86],
                   'SIZE':  ['L', 'XL', 'L', 'XL', 'M', 'S', 'M', 'L', 'L', 'XL'],
                   'COLOR': ['BL', 'RD', 'Y', 'GR', 'GR', 'RD', 'BL', 'Y', 'BL', 'GR'],
                   'class': ['A', 'C', 'B', 'B', 'C', 'A', 'A', 'A', 'C', 'C']},
                  columns=['DATE', 'NECK', 'BODY', 'SIZE', 'COLOR', 'class'])
df

df.isnull().sum()
df.dropna()
df.dropna(subset=['BODY'])
df.dropna(thresh=5)
df.fillna(df.mean())
df.fillna(0)
ip = df.interpolate(method='linear')
ip
size_mapping={'S':1,'M':2,'L':3,'XL':4}
df['SIZE']=df['SIZE'].map(size_mapping)
df

pd.get_dummies(df['COLOR'])
class_mapping = {'A':0,'B':1,'C':2}
class_mapping
df['class']=df['class'].map(class_mapping)
df
