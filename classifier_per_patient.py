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
    read_data = pd.read_csv(os.path.join(path_dataset, filename), encoding = "ISO-8859-1")[['point_awake', 'point_sleep', 'Accl Validity','dateTime']]
    
    #print(filename, len(read_data))
    df = pd.concat((df, read_data),ignore_index = True)


#initialize X and y
df['activity_level'] = ((df['point_awake'])/(df['point_sleep']+df['point_awake']))*100
#df = df[not np.isnan(df.activity_level)]
df = df[df['activity_level'].notna()]
df = df.reset_index()

count=0
dateTimeArr = ["" for x in range(0,len(df['Accl Validity']))]
for i in range(0,len(df['dateTime'])):    
    if ((df['Accl Validity'][i]!=0) and (df['activity_level'][i]==0)):
        dateTimeArr[count] = df['dateTime'][i]
        count+=1

X = df.reset_index()['activity_level']
logX = np.log(X)
y = df.reset_index()['Accl Validity']
y_binary1 = np.zeros(len(y))
y_binary2 = np.zeros(len(y))
X0 = df[df['Accl Validity']==0]['activity_level']
X1 = df[df['Accl Validity']==1]['activity_level']#.drop[0]
#X0 = df[df['Accl Validity']==0]['activity_level']#.drop[0]
X2 = df[df['Accl Validity']==2]['activity_level']#.drop[0]
#X2 = df[df['Accl Validity']==2].drop(df.iloc[0,:], inplace=True)['activity_level']
X1and2 = df[df['Accl Validity']!=0]['activity_level']


#remove 0 for log transformation and clearer distribution graph
X0wo0 = df[df['Accl Validity']==0].reset_index()['activity_level']
X1wo0 = df[df['Accl Validity']==1].reset_index()['activity_level']#.drop(index=[0,3,4,9,10,11,12,13,14])
X2wo0 = df[df['Accl Validity']==2].reset_index()['activity_level']#.drop(index=[0,1,2,3,4,5,6])
X1and2wo0 = df[df['Accl Validity']!=0].reset_index()['activity_level']#.drop(index=[0,1,4,5,6,7,8,9,10,11])


X0wo0 = X0wo0[(X0wo0!=0)]
X1wo0 = X1wo0[(X1wo0!=0)]
X2wo0 = X2wo0[(X2wo0!=0)]
X1and2wo0 = X1and2wo0[(X1and2wo0!=0)]

"""
X0wo0 = X0wo0[(['activity_level']<50)]
X1wo0 = X1wo0[(df['activity_level']<50)]
X2wo0 = X2wo0[df['activity_level']<50]
X1and2wo0 = X1and2wo0[(df['activity_level']<50)]
"""
binsno = 50

plt.figure(5)
plt.hist(X1and2,bins=binsno,histtype = 'step',color ='blue', label = 'Have motion')
plt.hist(X0,bins=binsno, color ='red',label = 'No motion')
plt.title("No motion and have motion distribution")
plt.xlabel('Activity Score')
plt.ylabel('Frequency')
plt.legend()

plt.figure(6)
plt.hist(X1,bins=binsno, histtype = 'step',color = 'blue',label = 'Moderate motion')
plt.hist(X2,bins=binsno, color = 'green',label = 'High motion')
plt.title("Moderate motion and High motion distribution")
plt.xlabel('Activity Score')
plt.ylabel('Frequency')
plt.legend()

plt.figure(7)
plt.hist(X1and2wo0,bins=binsno,histtype = 'step',color ='blue', label = 'Have motion')
plt.hist(X0wo0,bins=binsno, color ='red',label = 'No motion')
plt.title("No motion and have motion distribution")
plt.xlabel('Activity Score')
plt.ylabel('Frequency')
plt.legend()

plt.figure(8)
plt.hist(X1wo0,bins=binsno, histtype = 'step',color = 'blue',label = 'Moderate motion')
plt.hist(X2wo0,bins=binsno, color = 'green',label = 'High motion')
plt.title("Moderate motion and High motion distribution")
plt.xlabel('Activity Score')
plt.ylabel('Frequency')
plt.legend()

plt.figure(9)
plt.hist(np.log(X1and2wo0),bins=binsno, histtype = 'step',color = 'blue',label = 'Have motion')
plt.hist(np.log(X0wo0),bins=binsno, color = 'red',label = 'No motion')
plt.title("No motion and Have motion log-transformed distribution")
plt.xlabel('Activity Score')
plt.ylabel('Frequency')
plt.legend()

plt.figure(10)
plt.hist(np.log(X1wo0),bins=binsno, histtype = 'step',color = 'blue',label = 'Moderate motion')
plt.hist(np.log(X2wo0),bins=binsno, color = 'green',label = 'High motion')
plt.title("Moderate motion and High motion log-transformed distribution")
plt.xlabel('Activity Score')
plt.ylabel('Frequency')
plt.legend()


"""(+200)
for i in range(0,len(y)):
    if y[i] == 2:
        y_binary1[i] = 1

counterr=0
m=0
for i in range(0, len(y)-counterr):
    if y[i] ==0:
        y_binary2 = np.delete(y_binary2, i-counterr)
        counterr+=1
    else:
        if y[i] == 1:
            y_binary2[i-counterr] = 0
        else:
            y_binary2[i-counterr] = 1
n_1v2 = len(y_binary2)
#print(counterr)

#clf = OneVsRestClassifier(SVC()).fit(X, y)
#print(clf.score(X,y))

threshold_01 = np.arange(0,100,0.5)
threshold_12 = np.arange(0,100,0.5)

n = len(threshold_01)
y_2d = np.zeros((len(y), len(threshold_01)))

for i in range(0, len(y)):
    for j in range(0,n):
        y_2d[i][j] = y[i]

cm0 = np.zeros((len(threshold_01),3,3))
cm1 = np.zeros((int(n*(n-1)/2),3,3),)
cm2 = np.zeros((len(threshold_01),3,3))
cm0v1 = np.zeros((n,2,2))
cm1v2 = np.zeros((n_1v2,2,2))

delta_x = threshold_01[1] - threshold_01[0]
counter=0
y_pred0 = np.zeros(len(y))
y_pred1 = np.zeros(len(y))
y_pred2 = np.zeros(len(y))
y_pred = np.zeros(len(y))
y_pred02d = np.zeros((len(y), len(threshold_01)))
y_pred0v1 = np.zeros(len(y))
y_pred1v2 = np.zeros(len(y_binary2))


#print(X[0])
fp0 = np.empty(len(threshold_01), dtype = float)
tp0 = np.empty(len(threshold_01), dtype = float)
#fp1 =np.empty(int(n*(n-1)/2), dtype = float)
#tp1 = np.empty(int(n*(n-1)/2), dtype = float)
fp2 =np.empty(len(threshold_01), dtype = float)
tp2 = np.empty(len(threshold_01), dtype = float)
fn0 =np.empty(len(threshold_01), dtype = float)
tn0 =np.empty(len(threshold_01), dtype = float)
fn2 =np.empty(len(threshold_01), dtype = float)
tn2 =np.empty(len(threshold_01), dtype = float)
#fn1 =np.empty(int(n*(n-1)/2), dtype = float)
#tn1 = np.empty(int(n*(n-1)/2), dtype = float)

fp0v1 = np.empty(len(threshold_01), dtype = float)
tp0v1 = np.empty(len(threshold_01), dtype = float)
fp1v2 =np.empty(n, dtype = float)
tp1v2 = np.empty(n, dtype = float)
fn0v1 =np.empty(len(threshold_01), dtype = float)
tn0v1 =np.empty(len(threshold_01), dtype = float)
fn1v2 =np.empty(n, dtype = float)
tn1v2 = np.empty(n, dtype = float)

#for no motion vs have motion
counter=0
for i in threshold_01:
    for j in range(0, len(X)):
        if X[j]<=i:
            y_pred0v1[j] = 0
            #y_pred02d[j][isub] = 0
        else:
            y_pred0v1[j] = 1
            #y_pred02d[j][isub] = 1
    cm0v1[counter] = confusion_matrix(y_binary1, y_pred0v1)
    counter+=1

for i in range(0,len(threshold_01)):
    fn0v1[i] = cm0v1[i][1][0]
    tn0v1[i] = cm0v1[i][0][0]
    fp0v1[i] = cm0v1[i][0][1]
    tp0v1[i] = cm0v1[i][1][1]

fp0v1/=fp0v1.max()
tp0v1/=tp0v1.max()
fn0v1/=fn0v1.max()
tn0v1/=tn0v1.max()
sens0v1 = tp0v1/(fn0v1+tp0v1)
spec0v1 = tn0v1/(tn0v1+fp0v1)
#aa = metrics.multilabel_confusion_matrix(y_2d, y_pred02d)

fp0v1 = np.append(fp0v1, 0)
tp0v1 = np.append(tp0v1, 0)
sens0v1 = np.append(sens0v1, 0)
spec0v1 = np.append(spec0v1, 1)
sens0v1[0] = 1 
spec0v1[0] = 0 
sens1v2 = np.append(sens0v1, 0)
spec1v2 = np.append(spec0v1, 1)

plt.figure(3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
s="AUC = %.3f"
area0v1 = auc(fp0v1,tp0v1)
plt.plot(fp0v1,tp0v1, 'r-', label = 'no vs have motion')
plt.plot([], [], ' ', label=s%area0v1)
plt.title('ROC (TP vs FP)')
plt.legend()

#plt.figtext(0.7,0.2,s%area0v1)

plt.figure(4)

areass0v1 = auc(sens0v1,spec0v1)
plt.plot(spec0v1,sens0v1, 'r-', label = 'no vs have motion')
plt.plot([], [], ' ', label=s%areass0v1)
plt.xlabel('Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC (Sens vs Spec)')
plt.legend()


#plt.figtext(0.2,0.2,s%areass0v1)



print(area0v1)


#for moderate vs high motion
counter=0
for i in threshold_12:
    for j in range(0, n_1v2):
        if X[j]>=i:
            y_pred1v2[j] = 1
            #y_pred02d[j][isub] = 0
        else:
            y_pred1v2[j] = 0
            #y_pred02d[j][isub] = 1
    cm1v2[counter] = confusion_matrix(y_binary2, y_pred1v2)
    counter+=1

for i in range(0,len(threshold_12)):
    fn1v2[i] = cm1v2[i][1][0]
    tp1v2[i] = cm1v2[i][1][1]
    fp1v2[i] = cm1v2[i][0][1]
    tn1v2[i] = cm1v2[i][0][0]

fp1v2/=fp1v2.max()
tp1v2/=tp1v2.max()
fn1v2/=fn1v2.max()
tn1v2/=tn1v2.max()
sens1v2 = tp1v2/(fn1v2+tp1v2)
spec1v2 = tn1v2/(tn1v2+fp1v2)
#aa = metrics.multilabel_confusion_matrix(y_2d, y_pred02d)

fp1v2 = np.append(fp1v2, 0)
tp1v2 = np.append(tp1v2, 0)
spec1v2 = np.append(spec1v2,1)
sens1v2 = np.append(sens1v2,0)

plt.figure(3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
area1v2 = auc(fp1v2,tp1v2)
plt.plot(fp1v2,tp1v2, 'g-', label = 'moderate vs high motion')
plt.plot([], [], ' ', label=s%area1v2)
plt.title('ROC (TP vs FP)')
plt.legend()

plt.figure(4)
areass1v2 = auc(spec1v2,sens1v2)
plt.plot(spec1v2,sens1v2, 'g-', label = 'moderate vs high motion')
plt.plot([], [], ' ', label=s%areass1v2)
plt.xlabel('Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC (Sens vs Spec)')
plt.legend()



print(area1v2)


#distribution plot




#for lower threshold, 0-1
#isub=0
"""#(-200)
"""
counter=0
for i in threshold_01:
    for j in range(0, len(X)):
        if X[j]<i:
            y_pred0[j] = 0
            #y_pred02d[j][isub] = 0
        else:
            y_pred0[j] = 1
            #y_pred02d[j][isub] = 1
    cm0[counter] = confusion_matrix(y, y_pred0)
    counter+=1
    #isub+=1
    
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
#aa = metrics.multilabel_confusion_matrix(y_2d, y_pred02d)


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

"""
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

fp2 = np.append(fp2,0)
tp2 = np.append(tp2,0)

fn2 = np.append(fn2,1)
tn2 = np.append(tn2,1)

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
"""

"""(+49)
threshold_01 = np.append(threshold_01, 100)
spec0v1 = np.delete(spec0v1,100)
spec1v2 = np.delete(spec1v2,100)
sens1v2 = np.delete(sens1v2,100)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('threshold')
ax1.set_ylabel('sensitivity', color=color)
ax1.plot(sens0v1, threshold_01, color=color)
ax1.tick_params(axis='y', labelcolor=color)
plt.title('Sensitivity and Specificity vs 0-1 Activity threshold')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

threshold_01 = np.delete(threshold_01, 100)

color = 'tab:green'
ax2.set_ylabel('specificity', color=color)  # we already handled the x-label with ax1
ax2.plot(spec0v1, threshold_01, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


fig, ax1 = plt.subplots()
#threshold_01 = np.delete(threshold_01, 100)
threshold_12 = np.append(threshold_12, 100)

color = 'tab:red'
ax1.set_xlabel('threshold')
ax1.set_ylabel('sensitivity', color=color)
ax1.plot(sens1v2, threshold_01, color=color)
ax1.tick_params(axis='y', labelcolor=color)
plt.title('Sensitivity and Specificity vs 1-2 Activity threshold')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:green'
ax2.set_ylabel('specificity', color=color)  # we already handled the x-label with ax1
ax2.plot(spec1v2, threshold_01, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

"""#(-49)
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