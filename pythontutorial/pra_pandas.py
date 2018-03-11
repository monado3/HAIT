import numpy as np
import pandas as pd
from pandas import Series as se

# In[]: #Series
s1 = se([3, 7, 10, 13])
s1

a1 = np.array([3, 7, 10, 13])
a1

se(a1)

s1.values
s1.index

s2 = se([3, 7, 10, 13], index=['apple', 'orange', 'manago', 'peach'])
s2

s2[['apple', 'manago', 'orange']]

s2[s2 >= 5]

s2 >= 5

data = {'Tokyo': 1200, 'Sapporo': 800, 'Osaka': 1100, 'Okinawa': 200}

s3 = se(data)
s3

prefecture = ['Tokyo', 'Sapporo', 'Osaka', 'Okinawa', 'Fukuoka']
s4 = se(data, index=prefecture)
s4

pd.isnull(s4)
pd.notnull(s4)

# In[] #DataFrame
df = pd.DataFrame(
    {'名前': ['山田', '鈴木', '佐藤', '田中', '斉藤'],
     '年齢': [20, 34, 50, 12, 62],
     '性別': ['男', '男', '女', '男', '女']
     }
)

df

df1 = pd.DataFrame(
    {'名前': ['山田', '鈴木', '佐藤', '田中', '斉藤'],
     '年齢': [20, 34, 50, 12, 62],
     '性別': ['男', '男', '女', '男', '女']
     },
     columns=['名前','性別','年齢'],
     index=[1,2,3,4,5]
)

df1

data = pd.DataFrame( {'国・地域': {1: 'アメリカ合衆国', 2: 'ソビエト連邦', 3: '日本（開催国）', 4: '東西統一ドイツ', 5: 'イタリア', 6: 'ハンガリー', 7: 'ポーランド', 8: 'オーストラリア', 9: 'チェコスロバキア', 10: 'イギリス'},
 '計': {1: 90, 2: 96, 3: 29, 4: 50, 5: 27, 6: 22, 7: 23, 8: 18, 9: 14, 10: 18},
 '金': {1: 36, 2: 30, 3: 16, 4: 10, 5: 10, 6: 10, 7: 7, 8: 6, 9: 5, 10: 4},
 '銀': {1: 26, 2: 31, 3: 5, 4: 22, 5: 10, 6: 7, 7: 6, 8: 2, 9: 6, 10: 12},
 '銅': {1: 28, 2: 35, 3: 8, 4: 18, 5: 7, 6: 5, 7: 10, 8: 10, 9: 3, 10: 2},
 '順': {1: 'アメリカ合衆国',
          2: 'ソビエト連邦',
          3: '日本',
          4: '東西統一ドイツ',
          5: 'イタリア',
          6: 'ハンガリー',
          7: 'ポーランド',
          8: 'オーストラリア',
          9: 'チェコスロバキア',
          10: 'イギリス'}}
)

data

data.columns

data['順']
data2 = pd.DataFrame(data, columns=['国・地域', '金', '銀', '銅', '計'])
data2

data2.ix[3]
type(data2.ix[3])

data2.head()

data2.head(3)

data2.tail()

data3 = pd.DataFrame(data, columns=['国・地域', '金', '銀', '銅', '計', '選手数'])
data3

data3['選手数'] = np.arange(10)
data3

#data3['選手数'] = np.arange(9)
p = se(['200','300'], index=[2,5])
data3['選手数'] = p
data3

del data3['選手数']
data3

data3.drop(9, axis = 0)

data3.drop('金', axis=1)

data3.ix[1:6, :]

data3.ix[:5, 0:2]
