# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 18:33:15 2019

@author: 王思程
"""

import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
info=pd.read_excel('barrage2.xlsx')
info['text']=info['text'].astype(str)
info['happy']=pd.Series(np.zeros(len(info)))
info['favor_love']=pd.Series(np.zeros(len(info)))    
info['sorrow']=pd.Series(np.zeros(len(info)))    
info['angry']=pd.Series(np.zeros(len(info)))
info['scared']=pd.Series(np.zeros(len(info)))
info['hate']=pd.Series(np.zeros(len(info)))
info['surprised']=pd.Series(np.zeros(len(info)))
vecdic = pd.read_excel('finalver弹幕用情感词典.xlsx')
for i in range(len(info)):
    word_list=list(jieba.cut(info.ix[i,'text']))
    for word in word_list:
        if word in vecdic['word'].values:
            temp_df=vecdic.loc[ vecdic['word']==word ]
            temp_df.reset_index(drop=True, inplace=True)
            info.ix[i,'happy']+=temp_df.ix[0,'p1']
            info.ix[i,'favor_love']+=temp_df.ix[0,'p2']
            info.ix[i,'sorrow']+=temp_df.ix[0,'p3']
            info.ix[i,'angry']+=temp_df.ix[0,'p4']
            info.ix[i,'scared']+=temp_df.ix[0,'p5']
            info.ix[i,'hate']+=temp_df.ix[0,'p6']
            info.ix[i,'surprised']+=temp_df.ix[0,'p7']
info.sort_values(by='ordertime',ascending=True,inplace=True)
info.reset_index(drop=True, inplace=True)
info1=info[['happy','favor_love','sorrow','angry','scared','hate','surprised']]
info1.plot()
plt.show()


info.sort_values(by='realtime',ascending=True,inplace=True)
info.reset_index(drop=True, inplace=True)
info2=info[['happy','favor_love','sorrow','angry','scared','hate','surprised']]
for i in range(len(info2)):
    info2.ix[i,'motivation']=info2.ix[i,'happy']+info2.ix[i,'favor_love']-info2.ix[i,'sorrow']-info2.ix[i,'angry']-info2.ix[i,'scared']-info2.ix[i,'hate']

info2=info2['motivation'].cumsum()
info2.plot()
plt.show()


