# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:08:10 2019

@author: 王思程
"""

import pandas as pd
import numpy as np
import jieba
import math

df=pd.read_excel('barrage.xlsx')
#df=df[['id','ordertime','realtime','text']]
df['text']=df['text'].astype(str)
#df.reset_index(drop=True, inplace=True)
all_word=[]
all_word_id=[]
for i in range(len(df)):
    if '233' in df.ix[i,'text']:
        df.ix[i,'text']='23333'
    if '66' in df.ix[i,'text']:
        df.ix[i,'text']='666'
    if '哈哈' in df.ix[i,'text']:
        df.ix[i,'text']='哈哈'
    if 'hh' in df.ix[i,'text']:
        df.ix[i,'text']='hhh'
    seg_list = list(jieba.cut(df.ix[i,'text']))
    for word in seg_list:
        all_word.append(word)
        all_word_id.append(df.ix[i,'id'])
stopwords = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8').readlines()]
stopwords.extend(['！','，',' ','（','）','≦','≧','？','∇','～','に','⊙','の','は','：'])
word=[]
word_num=[]
word_inarticle=[]
word_TFIDF=[]
for i in range(len(all_word)):
    if all_word[i] not in word:
        if all_word[i] not in stopwords:
            word.append(all_word[i])
            word_num.append(1)
    else:
        pos=-1
        for j in range(len(word)):
            if word[j]==all_word[i]:
                pos=j
        word_num[pos]+=1

for i in range(len(word)):
    in_id=set()
    for j in range(len(all_word)):
        if word[i]==all_word[j]:
            in_id.add(all_word_id[j])
    word_inarticle.append(len(in_id))

TFIDF_df=pd.DataFrame({'word':word,'num':word_num,'inarticle':word_inarticle})
TFIDF_df=TFIDF_df[TFIDF_df['num']>5]
TFIDF_df=TFIDF_df[TFIDF_df['inarticle']>1]
TFIDF_df.reset_index(drop=True, inplace=True)
for i in range(len(TFIDF_df)):
    TFIDF_df.ix[i,'TFIDF']= TFIDF_df.ix[i,'num']/ len(all_word) *math.log(max(all_word_id),TFIDF_df.ix[i,'inarticle'])
TFIDF_df.sort_values(by=['TFIDF'],ascending=False,inplace=True)
TFIDF_df.reset_index(drop=True, inplace=True)
TFIDF_df_all=TFIDF_df
TFIDF_df=TFIDF_df[0:500]#选了弹幕词，根据情况后续细化改

#选的弹幕词和对应视频id号，存入danmuci.xlsx
danmu_word=[]
danmu_word_id=[]
for i in range(len(all_word)):
    if all_word[i] in TFIDF_df.loc[:,'word'].values :
        if all_word[i] not in ['666','23333','hhh','哈哈']:
            if len(all_word[i])>1:
                danmu_word.append(all_word[i])
                danmu_word_id.append(all_word_id[i])    
danmuci_df=pd.DataFrame({'word':danmu_word,'id':danmu_word_id})
danmuci_df=danmuci_df.drop_duplicates().reset_index(drop=True)
danmuci_df.to_excel('danmuci.xlsx')

#不在弹幕词且常用词典里有的词和对应id，存在yiyouci.xlsx
vecdic=pd.read_excel('vecdic.xlsx')
vecdic.reset_index(drop=True, inplace=True)
yiyou_word=[]
yiyou_word_id=[]
for i in range(len(all_word)):
    if all_word[i] not in danmu_word:
        if all_word[i] in vecdic.loc[:,'词语'].values:
            yiyou_word.append(all_word[i])
            yiyou_word_id.append(all_word_id[i])
        
yiyouci_df=pd.DataFrame({'word':yiyou_word,'id':yiyou_word_id})
yiyouci_df=yiyouci_df.drop_duplicates().reset_index(drop=True)
yiyouci_df.to_excel('yiyouci.xlsx')

#给这些词一个统一编号，存在cibianhao.xlsx
cibianhao_df=pd.concat([danmuci_df,yiyouci_df],ignore_index=True)
for i in range(len(cibianhao_df)):
    cibianhao_df.ix[i,'ci'] = i
    cibianhao_df.ix[i,'know']= cibianhao_df.ix[i,'word'] in yiyou_word
cibianhao_df.to_excel('cibianhao.xlsx')
