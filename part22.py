# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:00:18 2019

@author: 王思程
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:45:34 2019

@author: 王思程
"""
import pandas as pd
import numpy as np
import jieba
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
cibianhao_df=pd.read_excel('cibianhao.xlsx')
all_info_df=pd.read_excel('barrage.xlsx')
all_info_df=all_info_df[['id','ordertime','realtime','text']]
all_info_df['text']=all_info_df['text'].astype(str)
#all_info_df=all_info_df[all_info_df['id']%17==0]
all_info_df=all_info_df.sort_values(['id','ordertime'])
all_info_df.reset_index(drop=True, inplace=True)
vecdic_df=pd.read_excel('vecdic.xlsx')
#初始化label_sure_ur
label_sure_ur=cibianhao_df[cibianhao_df['know']==True]
label_sure_ur=label_sure_ur.drop(['id','know'], axis=1)
label_sure_ur.reset_index(drop=True, inplace=True)
label_sure_ur['p1']=pd.Series(np.zeros(len(label_sure_ur)))
label_sure_ur['p2']=pd.Series(np.zeros(len(label_sure_ur)))
label_sure_ur['p3']=pd.Series(np.zeros(len(label_sure_ur)))
label_sure_ur['p4']=pd.Series(np.zeros(len(label_sure_ur)))
label_sure_ur['p5']=pd.Series(np.zeros(len(label_sure_ur)))
label_sure_ur['p6']=pd.Series(np.zeros(len(label_sure_ur)))
label_sure_ur['p7']=pd.Series(np.zeros(len(label_sure_ur)))
for i in range(len(label_sure_ur)):
    word=cibianhao_df.ix[ label_sure_ur.ix[i,'ci'] , 'word']
    for j in range(len(vecdic_df)):
        if vecdic_df.ix[j,'词语'] == word:
            qinggan=vecdic_df.iloc[j,4]
            qingganqiangdu=vecdic_df.iloc[j,5] 
    if qinggan == 'PA' or qinggan=='PE':
        label_sure_ur.ix[i,'p1']+=qingganqiangdu
    elif qinggan=='PD' or qinggan=='PH' or qinggan=='PG' or qinggan=='PB' or qinggan=='PK':
        label_sure_ur.iloc[i,2]+=qingganqiangdu
    elif qinggan=='NA':
        label_sure_ur.ix[i,'p3']+=qingganqiangdu
    elif qinggan=='NB' or qinggan=='NJ' or qinggan=='NH' or qinggan=='PF':
        label_sure_ur.ix[i,'p4']+=qingganqiangdu
    elif qinggan=='NI' or qinggan=='NC' or qinggan=='NG':
        label_sure_ur.ix[i,'p5']+=qingganqiangdu
    elif qinggan =='PC':
        label_sure_ur.ix[i,'p7']+=qingganqiangdu
    else:
        label_sure_ur.ix[i,'p6']+=qingganqiangdu

label_sure_ur.to_excel('已知词.xlsx')

#写一个根据label_sure_ur和allinfo，词编号更新seq_sure,label_sure,seq_notsure的函数
#  1,得到【id,ordertime,realtime,弹幕内容,7维】
#  2，得到【id,inttime,弹幕分出来的词，7维】
#  3，遍历cibianhao，每一项是某格视频里的一个词，去上个表中
#     间找当前出现视频里的每一个这个词，这个词找20*7的数据
#     是know=True的话，还维label_sure加一行
seq_sure=pd.DataFrame(columns=['ci','p1','p2','p3','p4','p5','p6','p7'])
label_sure=pd.DataFrame(columns=['ci','p1','p2','p3','p4','p5','p6','p7'])
seq_notsure=pd.DataFrame(columns=['ci','p1','p2','p3','p4','p5','p6','p7'])
def renew(label_sure_ur):
    global seq_sure
    global label_sure
    global seq_notsure
    seq_sure.drop(seq_sure.index,inplace=True)
    label_sure.drop(label_sure.index,inplace=True)
    seq_notsure.drop(seq_notsure.index,inplace=True)
    temp1=all_info_df.copy()
    temp1['text']=temp1['text'].astype(str)
    temp1['p1']=pd.Series(np.zeros(len(temp1)))
    temp1['p2']=pd.Series(np.zeros(len(temp1)))    
    temp1['p3']=pd.Series(np.zeros(len(temp1)))    
    temp1['p4']=pd.Series(np.zeros(len(temp1)))
    temp1['p5']=pd.Series(np.zeros(len(temp1)))
    temp1['p6']=pd.Series(np.zeros(len(temp1)))
    temp1['p7']=pd.Series(np.zeros(len(temp1)))
    for i in range(len(temp1)):#对每条弹幕
        if '233' in temp1.ix[i,'text']:
            temp1.ix[i,'text']='23333'
        if '66' in temp1.ix[i,'text']:
            temp1.ix[i,'text']='666'
        if '哈哈' in temp1.ix[i,'text']:
            temp1.ix[i,'text']='哈哈'
        if 'hh' in temp1.ix[i,'text']:
            temp1.ix[i,'text']='hhh'
        seq_list=list(jieba.cut(temp1.ix[i,'text']))
        temp1.set_value(i, 'text', seq_list)
        for word in seq_list:#对该条弹幕里的每个词，在label_sure_ur里找到
            for j in range(len(label_sure_ur)):
                if word == label_sure_ur.ix[j,'word']:
                    temp1.ix[i,'p1']+=label_sure_ur.ix[j,'p1']
                    temp1.ix[i,'p2']+=label_sure_ur.ix[j,'p2']
                    temp1.ix[i,'p3']+=label_sure_ur.ix[j,'p3']
                    temp1.ix[i,'p4']+=label_sure_ur.ix[j,'p4']
                    temp1.ix[i,'p5']+=label_sure_ur.ix[j,'p5']
                    temp1.ix[i,'p6']+=label_sure_ur.ix[j,'p6']
                    temp1.ix[i,'p7']+=label_sure_ur.ix[j,'p7']
    temp2=temp1.drop('realtime',axis=1)
    temp2=temp2.rename(columns={'text':'word_list','ordertime':'inttime'})
    i=0#同一整数秒的弹幕合并

    while(i<len(temp2)-1):
        for j in range(1,len(temp2)):
            if i+j<len(temp2) and int(temp2.ix[i,'inttime'])==int(temp2.ix[i+j,'inttime']):
                temp2.ix[i,'inttime']=int(temp2.ix[i,'inttime'])
                temp2.ix[i,'word_list'].extend(temp2.ix[i+j,'word_list'])
                temp2.ix[i,'p1']+=temp2.ix[i+j,'p1']
                temp2.ix[i,'p2']+=temp2.ix[i+j,'p2']
                temp2.ix[i,'p3']+=temp2.ix[i+j,'p3']
                temp2.ix[i,'p4']+=temp2.ix[i+j,'p4']
                temp2.ix[i,'p5']+=temp2.ix[i+j,'p5']
                temp2.ix[i,'p6']+=temp2.ix[i+j,'p6']
                temp2.ix[i,'p7']+=temp2.ix[i+j,'p7']
            else:
                break
        i=i+j        
      
    temp2=temp2[temp2['inttime']%1==0]
    print('temp2\n')
    print(temp2.shape)
    print('\n')
#    print(temp2.head(5))

    #已完成情感背景还原，然后对cibianhao里每个词去找20*7
    for i in range(len(cibianhao_df)):#对某格视频里出现过的某个词
        id=cibianhao_df.ix[i,'id']
        word=cibianhao_df.ix[i,'word']
        ci=cibianhao_df.ix[i,'ci']
#        know=cibianhao_df.ix[i,'know']所以更新label_sure_ur后label_seq没变
        know=False
        for k in range(len(label_sure_ur)):
            if ci==label_sure_ur.ix[k,'ci']:
                know=True           
        temp3=temp2[temp2['id']==id]
        temp3.reset_index(drop=True, inplace=True)
#        print('temp3\n')
#        print(temp3.shape)
#        print('\n')
#        print('seq_sure')
#        print(seq_sure.shape)
#        print('label_sure')
#        print(label_sure.shape)
        for j in range(len(temp3)):
            if j>=10:
                if j<=len(temp3)-10:
                    if word in temp3.ix[j,'word_list']:   
                            temp4=temp3[j-10:j+10]
                            temp4=temp4.drop(['id','inttime','word_list'],axis=1)
                            temp4.reset_index(drop=True, inplace=True)
                            temp4['ci']=pd.Series(np.ones(20))
                            temp4['ci']*=ci
                            temp4_ci=temp4['ci']
                            temp4=temp4.drop('ci',axis=1)
                            temp4.insert(0,'ci',temp4_ci)
                            if know==True:
                                seq_sure=seq_sure.append(temp4,ignore_index=True)
                                temp5=label_sure_ur[label_sure_ur['ci']==ci]
                                temp5=temp5.drop('word',axis=1)
                                label_sure=label_sure.append(temp5,ignore_index=True)
                            else:
                                seq_notsure=seq_notsure.append(temp4,ignore_index=True)   
renew(label_sure_ur)              

#由seq_sure,label_sure,seq_notsure产生最终神经网络要学习的格式
def get_seq_array(seq_sure):
    for i in range(1,8):
        for j in range(len(seq_sure)):
            if seq_sure.ix[j,'p%d'%i]>20:
                seq_sure.ix[j,'p%d'%i]=1
            else:
                seq_sure.ix[j,'p%d'%i]=seq_sure.ix[j,'p%d'%i]/20
    seq_sure=seq_sure.drop('ci',axis=1)
    # 2维转3维 (samples, time steps, features) 
    def gen_sequence(df, seq_length, seq_cols):
        data_array = df[seq_cols].values
        for start in range(0,len(df),20):
            yield data_array[start:start+20,:]
    sequence_cols=['p1','p2','p3','p4','p5','p6','p7']
    seq_gen = list(gen_sequence(seq_sure, 20, sequence_cols))
    seq_array=np.concatenate([seq_gen],axis=0)
    return(seq_array)
seq_sure_array=get_seq_array(seq_sure)    
seq_notsure_array=get_seq_array(seq_notsure)
#print('seq_sure_array')
#print(seq_sure_array.shape)
#print('seq_notsure_array')
#print(seq_notsure_array.shape)

def get_label_array(label_sure):
    label_array=label_sure.drop('ci',axis=1)
    sequence_cols=['p1','p2','p3','p4','p5','p6','p7']
    label_array=label_array[sequence_cols].values
    return(label_array)    
label_sure_array=get_label_array(label_sure)   


#开始神经网络部分了
# build the network
sequence_length=20
nb_features = 7
nb_out = 7

model = Sequential()

model.add(LSTM(
         input_shape=(sequence_length, nb_features),
         units=64,
         return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
          units=64,
          return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(units=nb_out))

def my_loss(y_true,y_pred):
    loss=0
    y_true=K.eval(y_true)
    y_pred=K.eval(y_pred)
    for i in range(y_true.shape[0]):
        for j in range(7):
            if y_true[i][j]!=0:
                 loss+=K.square(y_true[i][j]-y_pred[i][j])
            else:
                 loss+=K.square(y_pred[i][j])*3
       
    return(loss)
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def my_acc(y_true,y_pred):
    acc=0
    y_true=K.eval(y_true)
    y_pred=K.eval(y_pred)
    for k in range(y_true.shape[0]):
        metrics=1
        for i in range(7):
            if y_true[0][i]==0 and y_pred[0][i]>3:
                metrics=0
            if y_true[0][i]!=0:
                 if y_true[0][i]-y_pred[0][i]>4:
                     metrics=0
                 if y_pred[0][i]-y_true[0][i]>4:
                     metrics=0
        acc+=metrics
    acc=acc/y_true.shape[0]        
    return(metrics)
def r_square(y_true, y_pred):   
    SSR = K.mean(K.square(y_pred-K.mean(y_true)),axis=-1)    
    SST = K.mean(K.square(y_true-K.mean(y_true)),axis=-1)    
    return SSR/SST
    
print('0 label_sure_array')
print(label_sure_array.shape)
print('0 seq_sure_array')
print(seq_sure_array.shape)

model.compile(loss='mse', optimizer='adam', metrics=[r_square])     
model.fit(seq_sure_array, label_sure_array, epochs=100, batch_size=300, validation_split=0.1, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')])
## training metrics
#scores = model.evaluate(seq_sure_array,label_sure_array, verbose=1, batch_size=32)
#print('Accurracy: {}'.format(scores[1]))   
#pre_array=model.predict(seq_sure_array)


#没训练好最后网络时：1，已有网络去预测seq_notsure_array
#                   2,处理得到label_notsure
#                   3,label_notsure分10块儿，分别加入label_sure_ur,更新seq_sure,用此时的网络去预测更新后的seq_sure，和label_sure对照求一个loss
#                   4，选loss最小的将对应的label_notsure加入  
while( len(label_sure_ur) < len(cibianhao_df)*0.8):                       
    label_notsure_array=model.predict(seq_notsure_array)
    sequence_cols=['p1','p2','p3','p4','p5','p6','p7']
    label_notsure=pd.DataFrame(label_notsure_array,columns=sequence_cols)
    for i in range(len(label_notsure)):
        j=(i+1)*20-1
        ci=seq_notsure.ix[j,'ci']
        label_notsure.ix[i,'ci']=ci
        label_notsure.ix[i,'word']=cibianhao_df.ix[ci,'word']
    mid = label_notsure['ci']
    label_notsure.drop(labels=['ci'], axis=1,inplace = True)
    label_notsure.insert(0, 'ci', mid)
    mid = label_notsure['word']
    label_notsure.drop(labels=['word'], axis=1,inplace = True)
    label_notsure.insert(0, 'word', mid)
    label_notsure_list=np.array_split(label_notsure,3,axis=0)
    loss=[]
    for i in range(3):
        part=label_notsure_list[i]
        print('label_sure_ur')
        print(label_sure_ur.shape)
        print('gengxinhou')
        print(label_sure_ur.append(part,ignore_index=True).drop_duplicates('ci').reset_index(drop=True).shape)
        renew(label_sure_ur.append(part,ignore_index=True).drop_duplicates('ci').reset_index(drop=True))
        seq_sure_array=get_seq_array(seq_sure) 
        label_sure_array=get_label_array(label_sure) 
        print('%d label_sure_array' %i)
        print(label_sure_array.shape)
        print('%d seq_sure_array' %i)
        print(seq_sure_array.shape)
         #防止选出的偏向情感值小的，因为新加的数量远小于原来的，所以出原有的一部分
#        print('1 label_sure_array')
#        print(label_sure_array.shape)
#        print('1 seq_sure_array')
#        print(seq_sure_array.shape)
#        print('part')
#        print(len(part))
#        label_sure_array=label_sure_array[len(label_sure_array)-3*len(part):len(label_sure_array)]
#        seq_sure_array=seq_sure_array[len(seq_sure_array)-3*len(part)*20:len(seq_sure_array)]
#        print('2 label_sure_array')
#        print(label_sure_array.shape)
#        print('2 seq_sure_array')
#        print(seq_sure_array.shape)
        model.fit(seq_sure_array, label_sure_array, epochs=100, batch_size=50, validation_split=0.05, verbose=1,
             callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
        scores = model.evaluate(seq_sure_array,label_sure_array, verbose=1, batch_size=32)
        loss.append(scores[0])
    pos=loss.index(min(loss))
    label_sure_ur=label_sure_ur.append(label_notsure_list[pos],ignore_index=True).drop_duplicates('ci').reset_index(drop=True)
    for ci in label_sure_ur['ci']:
        cibianhao_df.ix[ci,'know']=True
    renew(label_sure_ur)
    seq_sure_array=get_seq_array(seq_sure) 
    seq_notsure_array=get_seq_array(seq_notsure)
    label_sure_array=get_label_array(label_sure) 
    model.fit(seq_sure_array, label_sure_array, epochs=10, batch_size=32, validation_split=0.05, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])

    #训练好了把剩下的词也预测了，修改好格式，加入已知
label_notsure_array=model.predict(seq_notsure_array)
sequence_cols=['p1','p2','p3','p4','p5','p6','p7']
label_notsure=pd.DataFrame(label_notsure_array,columns=sequence_cols)
for i in range(len(label_notsure)):
    j=(i+1)*20-1
    ci=seq_notsure.ix[j,'ci']
    label_notsure.ix[i,'ci']=ci
    label_notsure.ix[i,'word']=cibianhao_df.ix[ci,'word']
mid = label_notsure['ci']
label_notsure.drop(labels=['ci'], axis=1,inplace = True)
label_notsure.insert(0, 'ci', mid)
mid = label_notsure['word']
label_notsure.drop(labels=['word'], axis=1,inplace = True)
label_notsure.insert(0, 'word', mid)  
label_sure_ur=label_sure_ur.append(label_notsure,ignore_index=True).drop_duplicates('ci').reset_index(drop=True)  
label_sure_ur[sequence_cols]=label_sure_ur[sequence_cols].astype(int)
label_sure_ur.sort_values(by='ci').to_excel('finalver弹幕用情感词典.xlsx')

pre_array=model.predict(seq_sure_array)
pre_array=pd.DataFrame(pre_array,columns=sequence_cols)
for i in range(len(pre_array)):
    j=(i+1)*20-1
    ci=seq_sure.ix[j,'ci']
    pre_array.ix[i,'ci']=ci
    pre_array.ix[i,'word']=cibianhao_df.ix[ci,'word']
pre_array=pre_array.drop_duplicates('ci').reset_index(drop=True)  
mid = pre_array['ci']
pre_array.drop(labels=['ci'], axis=1,inplace = True)
pre_array.insert(0, 'ci', mid)
mid = pre_array['word']
pre_array.drop(labels=['word'], axis=1,inplace = True)
pre_array.insert(0, 'word', mid)  
pre_array.to_excel('已有词的预测.xlsx')
    
    
    











    
    
    
    
    
    
    