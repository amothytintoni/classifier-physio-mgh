import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os 
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib.axis import Axis
from sklearn.metrics import confusion_matrix
from scipy.integrate import simps
from sklearn.metrics import auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import metrics

utc_offset = [-4, 0]
patient_id='154'

path_dataset = os.path.join('Dataset', patient_id)

ignored = {'undone'}
list_file = [x for x in os.listdir(path_dataset) if x not in ignored]
#print(list_file)​

df = pd.DataFrame()
for filename in list_file:
    # read_data = pd.read_excel(os.path.join(path_dataset, filename))
    #df = pd.read_csv('Dataset/154/154_2022-06-10.csv')[['point_awake', 'point_sleep', 'Accl Validity']]
    read_data = pd.read_csv(os.path.join(path_dataset, filename), encoding = "ISO-8859-1")[['point_awake', 'point_sleep', 'Accl Validity']]
    
    #print(filename, len(read_data))
    df = pd.concat((df, read_data),ignore_index = True)
    
df['activity_level'] = ((df['point_awake'])/(df['point_sleep']+df['point_awake']))*100
#df = df[not np.isnan(df.activity_level)]
df = df[df['activity_level'].notna()]
X = df.reset_index()['activity_level']
y = df.reset_index()['Accl Validity']

#clf = OneVsRestClassifier(SVC()).fit(X, y)
#print(clf.score(X,y))

threshold_01 = np.arange(0,100,0.5)
threshold_12 = np.arange(0,100,0.5)

n = len(threshold_01)
y_2d = np.zeros((len(y), len(threshold_01)))

for i in range(0, len(y)):
    for j in range(0,n):
        y_2d[i][j] = y[i]

cm0 = np.empty((len(threshold_01),3,3),dtype = float)
cm1 = np.empty((int(n*(n-1)/2),3,3),dtype = float)
cm2 = np.empty((len(threshold_01),3,3),dtype = float)
cm = np.empty((len(threshold_01),3,3),dtype = float)

delta_x = threshold_01[1] - threshold_01[0]
counter=0
y_pred0 = np.zeros(len(y))
y_pred1 = np.zeros(len(y))
y_pred2 = np.zeros(len(y))
y_pred = np.zeros(len(y))
y_pred02d = np.zeros((len(y), len(threshold_01)))


#print(X[0])
fp0 = np.empty(len(threshold_01), dtype = float)
tp0 = np.empty(len(threshold_01), dtype = float)
fp1 =np.empty(int(n*(n-1)/2), dtype = float)
tp1 = np.empty(int(n*(n-1)/2), dtype = float)
fp2 =np.empty(len(threshold_01), dtype = float)
tp2 = np.empty(len(threshold_01), dtype = float)
fn0 =np.empty(len(threshold_01), dtype = float)
tn0 =np.empty(len(threshold_01), dtype = float)
fn2 =np.empty(len(threshold_01), dtype = float)
tn2 =np.empty(len(threshold_01), dtype = float)
fn1 =np.empty(int(n*(n-1)/2), dtype = float)
tn1 = np.empty(int(n*(n-1)/2), dtype = float)


#for lower threshold, 0-1
isub=0
for i in threshold_01:
    for j in range(0, len(X)):
        if X[j]<=i:
            y_pred0[j] = 0
            y_pred02d[j][isub] = 0
        else:
            y_pred0[j] = 1
            y_pred02d[j][isub] = 1
    cm0[counter] = confusion_matrix(y, y_pred0)
    counter+=1
    isub+=1

"""
#    try
counter=0
for i in threshold_12:
    for j in range(0,len(X)):
        if X[j]> i:
            y_pred1[j] = 2
        
    cm1[counter] = confusion_matrix(y, y_pred1)
    counter+=1
#
for i in range(0,len(threshold_01)):
    for j in range(0,3):
        cm0[i][j][2] = cm1[i][j][2]
"""
    
for i in range(0,len(threshold_01)):
    fp0[i] = cm0[i][1][0] + cm0[i][2][0]
    tp0[i] = cm0[i][0][0]
    fn0[i] = cm0[i][0][1]+cm0[i][0][2]
    tn0[i] = cm0[i][1][1]+cm0[i][2][2]+cm0[i][1][2]+cm0[i][2][1]

fp0/=fp0.max()
tp0/=tp0.max()
fn0/=fn0.max()
tn0/=tn0.max()
sens0 = tp0/(fn0+tp0)
spec0 = tn0/(tn0+fp0)
aa = metrics.multilabel_confusion_matrix(y_2d, y_pred02d)


plt.figure(1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fp0,tp0, 'r-', label = '0-1 threshold')
plt.title('ROC (TP vs FP)')
plt.legend()

plt.figure(2)
plt.plot(spec0,sens0, 'r-', label = '0-1 threshold')
plt.xlabel('Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC (Sens vs Spec)')
plt.legend()

area0 = auc(fp0,tp0)


print(area0)


#for 1
"""
counter=0
​
k=0
​
for i in threshold_01:
    k = i + threshold_01[1] - threshold_01[0]
    while k<100:
        for j in range(0,len(X)):
            if X[j]< i:
                y_pred1[j] = 0
                continue
            if X[j]> k:
                y_pred1[j] = 2
                continue
            y_pred1[j] = 1
            
        cm1[counter] = confusion_matrix(y, y_pred1)
        counter+=1   
        k = k+ threshold_01[1] - threshold_01[0]
        
for i in range(0,int(n*(n-1)/2)):
    fp1[i] = cm1[i][0][1]+cm1[i][2][1]
    tp1[i] = cm1[i][1][1]
    #fn1[i] = cm1[i][1][0]+cm1[i][1][2]
    #tn1[i] = cm1[i][0][0]+cm1[i][2][0]+cm1[i][0][2] +cm1[i][2][2]
    
    
fp1/=fp1.max()
tp1/=tp1.max()
​
d= {'fp1': fp1, 'tp1': tp1}
df2 = pd.DataFrame(data = d)
df2 = df2.sort_values(by = ['fp1'])
df2 = df2.drop_duplicates(subset=['fp1'])
​
plt.figure(1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(df2['fp1'],df2['tp1'], 'b-', label = '1 activity')
plt.legend()
​
​
#area0 = auc(fp0,tp0)
area1 = auc(df2['fp1'],df2['tp1'])
​
#print(area0)
print(area1)
"""

#for 2 - higher threshold, between 1-2
counter=0
for i in threshold_12:
    for j in range(0, len(X)):
        if X[j]>=i:
            y_pred2[j] = 2
        else:
            y_pred2[j] = 1
    cm2[counter] = confusion_matrix(y, y_pred2)
    counter+=1
    
for i in range(0,len(threshold_01)):
    fp2[i] = cm2[i][0][2]+cm2[i][1][2]
    tp2[i] = cm2[i][2][2]
    fn2[i] = cm2[i][2][1]+cm2[i][2][0]
    tn2[i] = cm2[i][1][1]+cm2[i][0][0]+cm2[i][1][0]+cm2[i][0][1]

fp2/=fp2.max()
tp2/=tp2.max()
fn2/=fn2.max()
tn2/=tn2.max()
sens2 = tp2/(fn2+tp2)
spec2 = tn2/(tn2+fp2)

plt.figure(1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fp2,tp2, 'g-', label = '1-2 threshold')
plt.legend()

plt.figure(2)
plt.plot(spec2,sens2, 'g-', label = '1-2 threshold')
plt.xlabel('Specificity')
plt.ylabel('Sensitivity')
plt.legend()

area2 = auc(fp2,tp2)

print(area2)


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('threshold')
ax1.set_ylabel('sensitivity', color=color)
ax1.plot(sens0, threshold_01, color=color)
ax1.tick_params(axis='y', labelcolor=color)
plt.title('Sensitivity and Specificity vs 0-1 Activity threshold')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:green'
ax2.set_ylabel('specificity', color=color)  # we already handled the x-label with ax1
ax2.plot(spec0, threshold_01, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('threshold')
ax1.set_ylabel('sensitivity', color=color)
ax1.plot(sens2, threshold_01, color=color)
ax1.tick_params(axis='y', labelcolor=color)
plt.title('Sensitivity and Specificity vs 0-1 Activity threshold')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:green'
ax2.set_ylabel('specificity', color=color)  # we already handled the x-label with ax1
ax2.plot(spec2, threshold_01, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


"""
counter=0
for i in threshold_01:
    k = i + threshold_01[1]-threshold_01[0]
    
    if k>i:
        for j in range(0, len(X)):
            if X[j]>=i:
                if X[j]<= k:
                    y_pred1[j] = 1
                else:
                    y_pred1[j] = 2
            else:
                y_pred1[j] = 0
    cm1[counter] = confusion_matrix(y, y_pred1)
    counter+=1
    k+=1
​
​
for i in range(0,len(threshold_01)):
    fp1[i] = cm1[i][0][2]+cm1[i][1][2]
    tp1[i] = cm1[i][2][2]
    
​
fp1/=fp1.max()
tp1/=tp1.max()
​
plt.figure(2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fp1,tp1, 'b-', label = '1 activity')
plt.legend()
​
area1 = auc(fp1,tp1)
​
print(area1)
"""

"""
X0 = df[df['Accl Validity']==0].iloc[1:-1]['activity_level']
X1 = df[df['Accl Validity']==1].iloc[1:-1]['activity_level']#.drop[0]
#X0 = df[df['Accl Validity']==0]['activity_level']#.drop[0]
X2 = df[df['Accl Validity']==2].iloc[1:-1]['activity_level']#.drop[0]
#X2 = df[df['Accl Validity']==2].drop(df.iloc[0,:], inplace=True)['activity_level']
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
#X_train,X_val,y_train,y_val = train_test_split(X_train,y_train, test_size=0.2)
​
​
def build_model():
    model = Sequential()
    model.add(Dense(40, input_dim=1, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
"""
#keras_model = build_model()
#keras_model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

#X = np.delete(X,0)
"""
plt.figure(1)
fig, ax = plt.subplots() 
plt.axis([0,100,0,0.15])
plt.xlabel('Activity Score')
plt.ylabel('Frequency')
plt.hist(X0, bins=100, density=True, color='red')
plt.hist(X1, bins=100, density=True, color='cyan')
plt.hist(X2, bins=100, density=True, color='black')
"""
#plt.show()

#tpf = np.empty(len(threshold_01), dtype = float)
"""
for i in range(0,len(threshold_01)):
    fp0[i] = cm[i][1][0]+cm[i][2][0]
    tp0[i] = cm[i][0][0]
    fp1[i] = cm[i][0][1]+cm[i][2][1]
    
    tp1[i] = cm[i][1][1]
    #print(tp1[i])
    fp2[i] = cm[i][0][2]+cm[i][1][2]
    tp2[i] = cm[i][2][2]
    #tpf[i] = 1

fp1/=fp1.max()
fp2/=fp2.max()
fp0/=fp0.max()
tp0/=tp0.max()
tp1/=tp1.max()
tp2/=tp2.max()
#tpf/=tpf.max()
​
​
plt.figure(2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fp0,tp0, 'r-', label = '0 activity')
plt.plot(fp1,tp1,'b-', label = '1 activity')
plt.plot(fp2,tp2,'g-', label = '2 activity')
#plt.plot(threshold_12/100,tpf,'g-', label = 'f activity')
plt.legend()
​
area0 = auc(fp0,tp0)
#area1 = auc(fp1,tp1)
area2 = auc(fp2,tp2)
#area3 = simps(tpf, dx=0.005)
​
print(area0)
#print(area1)
print(area2)
#print(area3)
#print(fp0.max())
"""
#labels = df['Accl Validity']
#thres_low = 
#thres_hi =

#labels = [0,0,0,0,1,2,1,2,0,1,2,2,2,2] 
#preds =  [0,1,0,0,2,2,1,2,0,1,2,2,2,0]

#for i in range(0, df.shape[0]):
    #if