import MeCab as mc

# In[]
mecab = mc.Tagger()
text = '私は、東京都に住んでいます。'

print(mecab.parse(text))
# In[]
macab = mc.Tagger('-Owakati')
text = '私は、東京都に住んでいます。'
print(mecab.parse(text))

# In[]
