import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from matplotlib.axis import Axis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from scipy import stats
import sklearn.linear_model as sklm
import sklearn as sk
import statsmodels.formula.api as sm
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import feature_column
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing as prepro
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
#import correct_fnfp as corrector
import warnings
warnings.filterwarnings('ignore', '.*do not.*')

#import main_plot_ver2 as pp


#######################################################################
##########################################################################
#change accordingly
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.3f}".format

patient_grouping=True
samplefrac=1.0
trycount=1
utc_offset = [-4, 0]
# path_dataset = os.path.join('C:\\Users\\Timothy Antoni\\Documents\\','programming')
path_dataset = os.path.join('C:\\Users\\Timothy Antoni\\Documents\\programming\\Dataset', 'final1')
#path_dataset = os.path.join('C:\\Users\\Timothy Antoni\\Documents\\programming', 'fpfnplot')

buckets_count = 300 #for feature crossing bucketization

#enter any labelling correction file
#corr = pd.read_csv('Book1.csv')
#corr = corr.drop(columns = ['remark', 'motion validity'])

ignored = {'undone', 'sus', 'before_2', 'test','before_15_8'}
list_file = [x for x in os.listdir(path_dataset) if x not in ignored]
list_file = [x for x in list_file if x.endswith('.csv')]
#print(list_file)
######################################################################
####################################################################

#################
#### if got any correction (fn, fp)
# df = pd.read_csv(os.path.join(path_dataset,'df_corrected.csv'))
# df = df[['userId','dateTime', 
# 'PD Validity',                                                                                                'Accl Validity',
#     'point_awake', 'point_sleep',
#     # 'SQA_index', 
#     'accepted_frame_ratio', 
#     #'accepted_frame_tdomain_ratio', 
#     'list_sensor_onskin_status_ratio',
#     # 'val_sd_signal_w_sqa',
#     'list_onskin_status_stdev_ratio',
#         'skin_temperature',
#         'timeLengthValidity', 
#         ]]
############

df=pd.DataFrame()
for filename in list_file:
#     read_data = pd.read_csv(os.path.join(path_dataset, filename), encoding = "ISO-8859-1")[['userId', 'dateTime', 'point_awake', 'point_sleep', 
# 'Accl Validity',
# 'PD Validity',
# 'SQA_index', 
# 'accepted_frame_ratio', 
# 'accepted_frame_tdomain_ratio', 
# 'list_sensor_onskin_status_ratio',
# 'timeLengthValidity', 
# 'val_sd_signal_w_sqa', 
# 'val_sd_signal_wo_sqa', 
# 'list_onskin_status_stdev_ratio',
#     'median_list_output_kurtosis',
#     'median_list_output_peakratio',
#     'median_list_output_skewness',
#     'median_list_output_stdev',
#     'skin_temperature'
#     # 'RR_DC', 'RR_fdomain_w_good_sqa', 'RR_IBI', 'RR_TD', 'RR_tdomain', 'RR',
#     #'sensor_onskin_status'
#     ]]
    
    read_data = pd.read_csv(os.path.join(path_dataset, filename), encoding = "ISO-8859-1")[['userId','dateTime', 
'PD Validity',                                                                                                'Accl Validity',
    'point_awake', 'point_sleep',
    #'SQA_index', 
    'accepted_frame_ratio', 
    #'accepted_frame_tdomain_ratio', 
    'list_sensor_onskin_status_ratio',
    #'val_sd_signal_w_sqa',
    'list_onskin_status_stdev_ratio',
        'skin_temperature',
        'timeLengthValidity', 
    # 'RR_DC', 'RR_fdomain_w_good_sqa', 'RR_IBI', 'RR_TD', 'RR_tdomain', 'RR',
    #'sensor_onskin_status'
    ]]
    
    #print(filename, len(read_data))
    df = pd.concat((df, read_data),ignore_index = True)

# until here :)

#df = corrector.correct_fp(corr, df)

print(df.shape[0])
#############for all or df_4 df4
# feature_names = ['SQA_index', 
#                   'accepted_frame_ratio', 
#                   'accepted_frame_tdomain_ratio', 
#                   'list_sensor_onskin_status_ratio', 
#                   'val_sd_signal_w_sqa', 
#                   'val_sd_signal_wo_sqa', 
#                   'list_onskin_status_stdev_ratio',
#                   'median_list_output_kurtosis',
#                   'median_list_output_peakratio',
#                   'median_list_output_skewness',
#                   'median_list_output_stdev',
#                   'skin_temperature',
#                   'activity_level']

feature_names = [ #'SQA_index', 
                  'accepted_frame_ratio', 
                  #'accepted_frame_tdomain_ratio', 
                  'list_sensor_onskin_status_ratio',
                  # 'val_sd_signal_w_sqa',
                  'list_onskin_status_stdev_ratio',
                  'skin_temperature',
                  'activity_level'
                  ]
features = np.array(feature_names)



##drop nan na n/a values
for col in df.columns:
    df = df[df[col].notna()]

# for i in range(1080, 3000):
#     if df[df['recordReceivedByGateway']>=i].shape[0] !=6720:
#         print(i)
#         break
#print(df[df['recordReceivedByGateway']>=3000].shape[0])


df_nodrop = df
df = df.reset_index(drop=True)

#print(df['userId'][9993], df['dateT'])

df = df.sample(frac=samplefrac)
df = df.reset_index(drop=True)
print("all data=",df.shape[0])
df_nodrop = df_nodrop.sample(frac=samplefrac)


##############initial input 
##########################################################################
########################################################################







#df, df_test ,pdv_1_vs_all, pdv_1_vs_all_test = train_test_split(df, pdv_1_vs_all, test_size=0.15, shuffle=True)
# df, df_test ,pdv_1_vs_all, pdv_1_vs_all_test = train_test_split(df, pdv_1_vs_all, test_size=0.4, shuffle=True)
# df_nodrop, df_nodrop_test, pdvnd_1_vs_all, pdvnd_1_vs_all_test = train_test_split(df_nodrop, pdvnd_1_vs_all, test_size=0.4, shuffle=True)

###################################
#shuffle split data based on patients as each of them have unique characteristics
patient_ids = []

for i in df.index:
    if df['userId'][i] not in patient_ids:
        patient_ids = np.append(patient_ids, df['userId'][i])
print("no of patients", len(patient_ids))


df_train_ppt = pd.DataFrame() 
df_test_ppt = pd.DataFrame() 
df_nodrop_train_ppt = pd.DataFrame() 
df_nodrop_test_ppt = pd.DataFrame() 
for i in patient_ids:
    df_p1 = df[df['userId']==i]
    df_p1_train, df_p1_test = train_test_split(df_p1, test_size=0.4, shuffle=True)
    df_train_ppt = pd.concat((df_train_ppt,df_p1_train),ignore_index=True)
    df_test_ppt = pd.concat((df_test_ppt,df_p1_test),ignore_index=True)
    df_nodrop_train_ppt = pd.concat((df_nodrop_train_ppt,df_p1_train))
    df_nodrop_test_ppt = pd.concat((df_nodrop_test_ppt,df_p1_test))
    
if patient_grouping ==True:
    df = df_train_ppt
    df_test = df_test_ppt
    df_nodrop = df_nodrop_train_ppt
    df_nodrop_test = df_nodrop_test_ppt   

df = df.drop(columns=['userId','dateTime'])
df_nodrop = df_nodrop.drop(columns=['userId','dateTime'])
df_test = df_test.drop(columns=['userId','dateTime'])
df_nodrop_test = df_nodrop_test.drop(columns=['userId','dateTime'])

#################################################
#start of preprocessing

#initialize X and y
#print(df.shape)
#drop invalid length data
df = df[df['timeLengthValidity']==1].reset_index(drop=True)
df = df.drop(columns='timeLengthValidity')
df_nodrop = df_nodrop[df_nodrop['timeLengthValidity']==1]
df_nodrop = df_nodrop.drop(columns='timeLengthValidity')
df_test = df_test[df_test['timeLengthValidity']==1].reset_index(drop=True)
df_test = df_test.drop(columns='timeLengthValidity')
df_nodrop_test = df_nodrop_test[df_nodrop_test['timeLengthValidity']==1]
df_nodrop_test = df_nodrop_test.drop(columns='timeLengthValidity')
#print(df.shape)
#change RR-related columns to its relation with RR

# print("before rr cleaning", df.shape[0])
# df = df[df['RR_DC'].notna()]
# df = df[df['RR_fdomain_w_good_sqa'].notna()]
# df = df[df['RR_IBI'].notna()]
# df = df[df['RR_TD'].notna()]
# df = df[df['RR_tdomain'].notna()]
# df = df[df['RR'].notna()]
# print("after rr cleaning", df.shape[0])

# df['RR_DC'] = np.abs((df['RR_DC'] - df['RR'])/df['RR'])
# df['RR_fdomain_w_good_sqa'] = np.abs((df['RR_fdomain_w_good_sqa'] - df['RR'])/df['RR'])
# df['RR_IBI'] = np.abs((df['RR_IBI'] - df['RR'])/df['RR'])
# df['RR_TD'] = np.abs((df['RR_TD'] - df['RR'])/df['RR'])
# df['RR_tdomain'] = np.abs((df['RR_tdomain'] - df['RR'])/df['RR'])

# df = df.drop(columns='RR')

#clean n/a values
##############################
######activity_level
df['activity_level'] = ((df['point_awake'])/(df['point_sleep']+df['point_awake']))
df_nodrop['activity_level'] = ((df_nodrop['point_awake'])/(df_nodrop['point_sleep']+df_nodrop['point_awake']))
df_test['activity_level'] = ((df_test['point_awake'])/(df_test['point_sleep']+df_test['point_awake']))
df_nodrop_test['activity_level'] = ((df_nodrop_test['point_awake'])/(df_nodrop_test['point_sleep']+df_nodrop_test['point_awake']))
#print(df.shape)
#activity_level
#############################

# pd0 = df[df['PD Validity']==0].reset_index()['PD Validity']
# #print(len(pd0))
# pd1 = df[df['PD Validity']==1].reset_index()['PD Validity']
# #print(len(pd1))
# pd2 = df[df['PD Validity']==2].reset_index()['PD Validity']
#print(len(pd2))
# df = df[df['activity_level'].notna()]
# #print(df.shape)
# df = df[df['PD Validity'].notna()]
# #print(df.shape)
# df = df[df['SQA_index'].notna()]
# #print(df.shape)
# df = df[df['Accl Validity'].notna()]
# #print(df.shape)
# df = df[df['accepted_frame_ratio'].notna()]
# #print(df.shape)
# df = df[df['accepted_frame_tdomain_ratio'].notna()]
# #print(df.shape)
df = df.reset_index(drop=True)

pdv = df.reset_index()['PD Validity']
pdv_test = df_test['PD Validity']
pdvnd = df_nodrop['PD Validity']
pdvnd_test = df_nodrop_test['PD Validity']

pdv_1_vs_all = pdv
pdv_0_vs_all = pdv
pdvnd_1_vs_all = pdvnd
pdvnd_0_vs_all = pdvnd
pdv_1_vs_all_test = pdv_test
pdv_0_vs_all_test = pdv_test
pdvnd_1_vs_all_test = pdvnd_test
pdvnd_0_vs_all_test = pdvnd_test

for i in range (0,len(pdv)):
    if pdv[i]==2:
        pdv_1_vs_all[i] = 0
        pdv_0_vs_all[i] = 1

for i in range (0,len(pdv_test)):
    if pdv_test[i]==2:
        pdv_1_vs_all_test[i] = 0
        pdv_0_vs_all_test[i] = 1

for i in pdvnd.index:
    if pdvnd[i]==2:
        pdvnd_1_vs_all[i] = 0
        pdvnd_0_vs_all[i] = 1

for i in pdvnd_test.index:
    if pdvnd_test[i]==2:
        pdvnd_1_vs_all_test[i] = 0
        pdvnd_0_vs_all_test[i] = 1
# pdv_1_vs_all, pdv_1_vs_all_test = train_test_split(pdv_1_vs_all, test_size=0.4, shuffle=True)
# pdvnd_1_vs_all, pdvnd_1_vs_all_test = train_test_split(pdvnd_1_vs_all, test_size=0.4, shuffle=True)

print("train pd0 =",len(df[df['PD Validity']==0]))
print("train pd0 =",len(df[df['PD Validity']==1]))
print("train pd0 =",len(df[df['PD Validity']==2]))
print("test pd0 =",len(df_test[df_test['PD Validity']==0]))
print("test pd0 =",len(df_test[df_test['PD Validity']==1]))
print("test pd0 =",len(df_test[df_test['PD Validity']==2]))


# accf = df.reset_index(drop=True)['accepted_frame_ratio']
# #acct = df.reset_index(drop=True)['accepted_frame_tdomain_ratio']
# accf1 = df[df['PD Validity']==1].reset_index(drop=True)['accepted_frame_ratio']
# accf0 = df[df['PD Validity']==0].reset_index(drop=True)['accepted_frame_ratio']
#acct1 = df[df['PD Validity']==1].reset_index(drop=True)['accepted_frame_tdomain_ratio']
#acct0 = df[df['PD Validity']==0].reset_index(drop=True)['accepted_frame_tdomain_ratio']

# =============================================================================
# df['sqa_x_acc_time'] = df['SQA_index'] * df['accepted_frame_tdomain_ratio']
# df['sqa_x_acc_freq'] = df['SQA_index'] * df['accepted_frame_ratio']
# df_test['sqa_x_acc_time'] = df_test['SQA_index'] * df_test['accepted_frame_tdomain_ratio']
# df_test['sqa_x_acc_freq'] = df_test['SQA_index'] * df_test['accepted_frame_ratio']
# acctxsqa = df.reset_index(drop=True)['sqa_x_acc_freq']
# accfxsqa = df.reset_index(drop=True)['sqa_x_acc_time']
# =============================================================================

###################################################
###################################################
#1 initialize variables
##A. ACCL DECLARATIONS


##B. SQA DECLARATIONS

#sqa_index = df.reset_index()['SQA_index']
#threshold_sqa = np.arange(0,1,0.01)
threshold_sqa = np.arange(-0.01,1.01,0.01)
pdv_1_vs_all_pred = np.zeros(len(pdv_1_vs_all))
pdv_0_vs_all_pred = np.zeros(len(pdv_1_vs_all))
pdv_2_vs_all_pred = np.zeros(len(pdv_1_vs_all))
cmsqa = np.zeros((len(threshold_sqa),2,2))
cmsqa_0_vs_all = np.zeros((len(threshold_sqa),2,2))
cmsqa_1_vs_all = np.zeros((len(threshold_sqa),2,2))
#sqa0 = df[df['PD Validity']==0].reset_index(drop=True)['SQA_index']
#sqa1 = df[df['PD Validity']==1].reset_index(drop=True)['SQA_index']

# feature crossing delayed  =============================================================================
# boundaries = np.linspace(0,1,200)
# bucketized_accf = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(str(accf.to_numpy())), boundaries=boundaries)
# bucketized_acct = tf.feature_column.bucketized_column(acct, boundaries)
# bucketized_sqa = tf.feature_column.bucketized_column(sqa_index, boundaries)
#=============================================================================
# 
# 

# =============================================================================

#generate binary comparison data



##C. TOTAL_ACCEPTED_FRAME DECLARATIONS
# threshold_accf = np.arange(-0.01,1.01,0.01)
# pdv_1_vs_all_accf_pred = np.zeros(len(df['accepted_frame_ratio']))
# cm_accf = np.zeros((len(threshold_accf),2,2))

# = np.arange(-0.01,1.01,0.01)
#pdv_1_vs_all_acct_pred = np.zeros(len(df['accepted_frame_tdomain_ratio']))
#cm_acct = np.zeros((len(threshold_acct),2,2))

##D. ACC_FRAME X SQA_INDEX DECLARATIONS

# threshold_accf_x_sqa = np.arange(-0.01,1.01,0.01)
# pdv_1_vs_all_accfxsqa_pred = np.zeros(len(df['accepted_frame_ratio']))
# cm_accfxsqa = np.zeros((len(threshold_accf_x_sqa),2,2))

#threshold_acct_x_sqa = np.arange(-0.01,1.01,0.01)
#pdv_1_vs_all_acctxsqa_pred = np.zeros(len(df['accepted_frame_tdomain_ratio']))
#cm_acctxsqa = np.zeros((len(threshold_acct_x_sqa),2,2))

##D. ML related declarations
#df_predictor_1 = df[[#'SQA_index', 
                     #'accepted_frame_ratio','accepted_frame_tdomain_ratio']].reset_index(drop=True)
# df_2 = df_predictor_1.iloc[:,0:2]
#df_3 = df_predictor_1[[#'SQA_index', 
                       #'accepted_frame_tdomain_ratio']]
df_4 = df.drop(columns=df.iloc[:,0:4])
df_4_nodrop = df_nodrop.drop(columns=df.iloc[:,0:4])
#df_test_predictor_1 = df_test[[#'SQA_index',
                               #'accepted_frame_ratio','accepted_frame_tdomain_ratio']].reset_index(drop=True)
# df_test_2 = df_test_predictor_1.iloc[:,0:2]
#df_test_3 = df_test_predictor_1[[#'SQA_index', 
                                 #'accepted_frame_tdomain_ratio']]
df_test_4 = df_test.drop(columns=df_test.iloc[:,0:4])
df_test_4_nodrop = df_nodrop_test.drop(columns=df.iloc[:,0:4])

df_ML_results = pd.DataFrame()
df_fn_results = pd.DataFrame()
df_fp_results = pd.DataFrame()

##feature selection

# print("before selection 1")
# selector1 = VarianceThreshold(threshold=(.9*(1-.9)))
# df_4 = selector1.fit_transform(df_4)
# print("after selection 1", selector1.get_feature_names_out(selector1.feature_names_in_))

# df_test_4 = selector1.fit_transform(df_test_4)


################


print("train data length =", df.shape[0])
print("test data length =", df_test.shape[0])


# =============================================================================
    #preprocessing prepro
    
#sqa_numpy = sqa_index.to_numpy(copy=True)
#sqa_reshaped = np.reshape(sqa_numpy,(-1,1))
pdv_1_vs_all_numpy = pdv_1_vs_all.to_numpy(copy=True)
pdv_1_vs_all_reshaped = np.reshape(pdv_1_vs_all_numpy,(-1,1))
# accf_numpy = accf.to_numpy(copy=True)
# accf_reshaped = np.reshape(accf_numpy,(-1,1))
# acct_numpy = acct.to_numpy(copy=True)
# acct_reshaped = np.reshape(acct_numpy,(-1,1))
 
#sqa_test_numpy = df_test['SQA_index'].reset_index(drop=True).to_numpy(copy=True)
#sqa_test_reshaped = np.reshape(sqa_test_numpy,(-1,1))
pdv_1_vs_all_test_numpy = pdv_1_vs_all_test.to_numpy(copy=True)
pdv_1_vs_all_test_reshaped = np.reshape(pdv_1_vs_all_test_numpy,(-1,1))

# accf_test = df_test.reset_index(drop=True)['accepted_frame_ratio']
# #acct_test = df_test.reset_index(drop=True)['accepted_frame_tdomain_ratio']
# accf_test_numpy = accf_test.to_numpy(copy=True)
# accf_test_reshaped = np.reshape(accf_test_numpy,(-1,1))
# acct_test_numpy = acct_test.to_numpy(copy=True)
# acct_test_reshaped = np.reshape(acct_test_numpy,(-1,1))

#sqa_reshaped_scaled = MinMaxScaler().fit_transform(sqa_reshaped)
# accf_reshaped_scaled = MinMaxScaler().fit_transform(accf_reshaped)
# #acct_reshaped_scaled = MinMaxScaler().fit_transform(acct_reshaped)
# accf_x_sqa_reshaped_scaled = MinMaxScaler().fit_transform(df_2)
# acct_x_sqa_reshaped_scaled = MinMaxScaler().fit_transform(df_3)
# accft_x_sqa_reshaped_scaled = MinMaxScaler().fit_transform(df_predictor_1)
all_reshaped_scaled = MinMaxScaler().fit_transform(df_4)

#sqa_test_reshaped_scaled = MinMaxScaler().fit_transform(sqa_test_reshaped)
# accf_test_reshaped_scaled = MinMaxScaler().fit_transform(accf_test_reshaped)
# # acct_test_reshaped_scaled = MinMaxScaler().fit_transform(acct_test_reshaped)
# accf_x_sqa_test_reshaped_scaled = MinMaxScaler().fit_transform(df_test_2)
# acct_x_sqa_test_reshaped_scaled = MinMaxScaler().fit_transform(df_test_3)
# accft_x_sqa_test_reshaped_scaled = MinMaxScaler().fit_transform(df_test_predictor_1)
all_test_reshaped_scaled = MinMaxScaler().fit_transform(df_test_4)

#sqa_reshaped = sqa_reshaped_scaled
# accf_reshaped = accf_reshaped_scaled
# # acct_reshaped = acct_reshaped_scaled
# accf_x_sqa_reshaped_ = accf_x_sqa_reshaped_scaled
# acct_x_sqa_reshaped = acct_x_sqa_reshaped_scaled
# accft_x_sqa_reshaped = accft_x_sqa_reshaped_scaled
all_reshaped = all_reshaped_scaled

#sqa_test_reshaped = sqa_test_reshaped_scaled
# accf_test_reshaped = accf_test_reshaped_scaled
# # acct_test_reshaped = acct_test_reshaped_scaled
# accf_x_sqa_test_reshaped = accf_x_sqa_test_reshaped_scaled
# acct_x_sqa_test_reshaped = acct_x_sqa_test_reshaped_scaled
# accft_x_sqa_test_reshaped = accft_x_sqa_test_reshaped_scaled
all_test_reshaped = all_test_reshaped_scaled

df_4 = MinMaxScaler().fit_transform(df_4)
df_test_4 = MinMaxScaler().fit_transform(df_test_4)
df_4_nodrop = MinMaxScaler().fit_transform(df_4_nodrop)
df_test_4_nodrop = MinMaxScaler().fit_transform(df_test_4_nodrop)

# bucketized_accf = np.array_split(accf_numpy,buckets_count)
# bucketized_acct = np.array_split(acct_numpy,buckets_count)
#bucketized_sqa = np.array_split(sqa_numpy,buckets_count)



   # (-219)=============================================================================

   # =============================================================================
   # accf_x_sqa_numpy = accfxsqa.to_numpy(copy=True)
   # accf_x_sqa_reshaped = np.reshape(accf_x_sqa_numpy, (-1,1))
   # acct_x_sqa_numpy = acctxsqa.to_numpy(copy=True)
   # acct_x_sqa_reshaped = np.reshape(acct_x_sqa_numpy, (-1,1))
   # accft_x_sqa_numpy = accfxsqa.to_numpy(copy=True)
   # accft_x_sqa_reshaped = np.reshape(accft_x_sqa_numpy, (-1,1))
   # 
   # =============================================================================
   

###################################################
###################################################
#2 fill in the cm
    #a. PD 
#(+219) =============================================================================
# counter=0
# for i in threshold_sqa:
#     for j in range(0, len(pdv_1_vs_all)):
#         if (sqa_index[j] <= i):
#             pdv_1_vs_all_pred_0_vs_all[j] = 0
#             pdv_1_vs_all_pred_1_vs_all[j] = 0
#         else:
#             pdv_1_vs_all_pred_0_vs_all[j] = 1
#             pdv_1_vs_all_pred_1_vs_all[j] = 1
#            
#     cmsqa_0_vs_all[counter] = confusion_matrix(pdv_1_vs_all_0_vs_all, pdv_1_vs_all_pred_0_vs_all)
#     cmsqa_1_vs_all[counter] = confusion_matrix(pdv_1_vs_all_old, pdv_1_vs_all_pred_1_vs_all)
#     counter+=1
# 
# counter=0
# fnsqa_0_vs_all = np.empty(len(threshold_sqa), dtype = float)
# tnsqa_0_vs_all = np.empty(len(threshold_sqa), dtype = float)
# fpsqa_0_vs_all = np.empty(len(threshold_sqa), dtype = float)
# tpsqa_0_vs_all = np.empty(len(threshold_sqa), dtype = float)
# fnsqa_1_vs_all = np.empty(len(threshold_sqa), dtype = float)
# tnsqa_1_vs_all = np.empty(len(threshold_sqa), dtype = float)
# fpsqa_1_vs_all = np.empty(len(threshold_sqa), dtype = float)
# tpsqa_1_vs_all = np.empty(len(threshold_sqa), dtype = float)
# 
# for i in range(0, len(threshold_sqa)):
#     fpsqa_0_vs_all[i] = cmsqa_0_vs_all[i][1][0]#/len(pdv_1_vs_all)
#     tpsqa_0_vs_all[i] = cmsqa_0_vs_all[i][0][0]#/len(pdv_1_vs_all)
#     fnsqa_0_vs_all[i] = cmsqa_0_vs_all[i][0][1]#/len(pdv_1_vs_all)
#     tnsqa_0_vs_all[i] = cmsqa_0_vs_all[i][1][1]#/len(pdv_1_vs_all)
#     fnsqa_1_vs_all[i] = cmsqa_1_vs_all[i][1][0]#/len(pdv_1_vs_all)
#     tnsqa_1_vs_all[i] = cmsqa_1_vs_all[i][0][0]#/len(pdv_1_vs_all)
#     fpsqa_1_vs_all[i] = cmsqa_1_vs_all[i][0][1]#/len(pdv_1_vs_all)
#     tpsqa_1_vs_all[i] = cmsqa_1_vs_all[i][1][1]#/len(pdv_1_vs_all)
# 
# count_1_in_1_vs_all = np.empty(1,dtype=float)
# count_0_in_1_vs_all = np.empty(1,dtype=float)
# count_0_in_0_vs_all = np.empty(1,dtype=float)
# count_1_in_0_vs_all = np.empty(1,dtype=float)
# 
# for i in range(0, len(pdv_1_vs_all)):
#     if i==0:
#         count_1_in_1_vs_all = np.delete(count_1_in_1_vs_all, 0)
#         count_0_in_1_vs_all = np.delete(count_0_in_1_vs_all, 0)
#         count_0_in_0_vs_all = np.delete(count_0_in_0_vs_all, 0)
#         count_1_in_0_vs_all = np.delete(count_1_in_0_vs_all, 0)
#         
#     if pdv_1_vs_all[i] == 0:
#         count_0_in_1_vs_all = np.append(count_0_in_1_vs_all, i)
#         count_0_in_0_vs_all = np.append(count_0_in_0_vs_all, i)
#     if pdv_1_vs_all[i] == 1:
#         count_1_in_1_vs_all = np.append(count_1_in_1_vs_all, i)
#         count_1_in_0_vs_all = np.append(count_1_in_0_vs_all, i)
# 
# fnsqa_1_vs_all/=len(count_1_in_1_vs_all)#/=fnsqa_1_vs_all.max()
# fpsqa_1_vs_all/=len(count_0_in_1_vs_all)#/=fpsqa_1_vs_all.max()
# tnsqa_1_vs_all/=len(count_0_in_1_vs_all)#/=tnsqa_1_vs_all.max()
# tpsqa_1_vs_all/=len(count_1_in_1_vs_all)#/=tpsqa_1_vs_all.max()
# 
# fnsqa_0_vs_all/=len(count_1_in_0_vs_all)#/=fnsqa_0_vs_all.max()
# fpsqa_0_vs_all/=len(count_0_in_0_vs_all)#/=fpsqa_0_vs_all.max()
# tnsqa_0_vs_all/=len(count_0_in_0_vs_all)#/=tnsqa_0_vs_all.max()
# tpsqa_0_vs_all/=len(count_1_in_0_vs_all)#/=tpsqa_0_vs_all.max()
# 
# #d. accepted_frame  
# counter=0
# for i in threshold_accf:
#     for j in range(0, len(pdv_1_vs_all)):
#         if (accf[j] <= i):
#             pdv_1_vs_all_accf_pred[j] = 0
#         else:
#             pdv_1_vs_all_accf_pred[j] = 1
#     cm_accf[counter] = confusion_matrix(pdv_1_vs_all_old, pdv_1_vs_all_accf_pred)
#     counter+=1
#             
# counter=0
# for i in threshold_acct:
#     for j in range(0, len(pdv_1_vs_all)):
#         if (acct[j] <= i):
#             pdv_1_vs_all_acct_pred[j] = 0
#         else:
#             pdv_1_vs_all_acct_pred[j] = 1
#            
#     cm_acct[counter] = confusion_matrix(pdv_1_vs_all_old, pdv_1_vs_all_acct_pred)
#     counter+=1
# 
# counter=0
# fn_accf = np.empty(len(threshold_accf), dtype = float)
# tn_accf = np.empty(len(threshold_accf), dtype = float)
# fp_accf = np.empty(len(threshold_accf), dtype = float)
# tp_accf = np.empty(len(threshold_accf), dtype = float)
# fn_acct = np.empty(len(threshold_acct), dtype = float)
# tn_acct = np.empty(len(threshold_acct), dtype = float)
# fp_acct = np.empty(len(threshold_acct), dtype = float)
# tp_acct = np.empty(len(threshold_acct), dtype = float)
# 
# for i in range(0, len(threshold_sqa)):
#     fn_accf[i] = cm_accf[i][1][0]#/len(pdv_1_vs_all)
#     tn_accf[i] = cm_accf[i][0][0]#/len(pdv_1_vs_all)
#     fp_accf[i] = cm_accf[i][0][1]#/len(pdv_1_vs_all)
#     tp_accf[i] = cm_accf[i][1][1]#/len(pdv_1_vs_all)
#     fn_acct[i] = cm_acct[i][1][0]#/len(pdv_1_vs_all)
#     tn_acct[i] = cm_acct[i][0][0]#/len(pdv_1_vs_all)
#     fp_acct[i] = cm_acct[i][0][1]#/len(pdv_1_vs_all)
#     tp_acct[i] = cm_acct[i][1][1]#/len(pdv_1_vs_all)
# 
# count_1_in_acct = np.empty(1,dtype=float)
# count_0_in_acct = np.empty(1,dtype=float)
# count_0_in_accf = np.empty(1,dtype=float)
# count_1_in_accf = np.empty(1,dtype=float)
# 
# for i in range(0, len(pdv_1_vs_all)):
#     if i==0:
#         count_1_in_accf = np.delete(count_1_in_accf, 0)
#         count_0_in_accf = np.delete(count_0_in_accf, 0)
#         count_0_in_acct = np.delete(count_0_in_acct, 0)
#         count_1_in_acct = np.delete(count_1_in_acct, 0)
#         
#     if pdv_1_vs_all[i] == 0:
#         count_0_in_accf = np.append(count_0_in_accf, i)
#         count_0_in_acct = np.append(count_0_in_acct, i)
#     if pdv_1_vs_all[i] == 1:
#         count_1_in_accf = np.append(count_1_in_accf, i)
#         count_1_in_acct = np.append(count_1_in_acct, i)
# 
# fn_accf/=len(count_1_in_accf)#/=fnsqa_accf.max()
# fp_accf/=len(count_0_in_accf)#/=fpsqa_accf.max()
# tn_accf/=len(count_0_in_accf)#/=tnsqa_accf.max()
# tp_accf/=len(count_1_in_accf)#/=tpsqa_accf.max()
# 
# fn_acct/=len(count_1_in_acct)#/=fnsqa_acct.max()
# fp_acct/=len(count_0_in_acct)#/=fpsqa_acct.max()
# tn_acct/=len(count_0_in_acct)#/=tnsqa_acct.max()
# tp_acct/=len(count_1_in_acct)#/=tpsqa_acct.max()
# 
# ##e. sqa x acc frame vs PD
# counter=0
# for i in threshold_accf_x_sqa:
#     for j in range(0, len(pdv_1_vs_all)):
#         if (accfxsqa[j] <= i):
#             pdv_1_vs_all_accfxsqa_pred[j] = 0
#         else:
#             pdv_1_vs_all_accfxsqa_pred[j] = 1
#     cm_accfxsqa[counter] = confusion_matrix(pdv_1_vs_all_old, pdv_1_vs_all_accfxsqa_pred)
#     counter+=1
#             
# counter=0
# for i in threshold_acct_x_sqa:
#     for j in range(0, len(pdv_1_vs_all)):
#         if (acctxsqa[j] <= i):
#             pdv_1_vs_all_acctxsqa_pred[j] = 0
#         else:
#             pdv_1_vs_all_acctxsqa_pred[j] = 1
#            
#     cm_acctxsqa[counter] = confusion_matrix(pdv_1_vs_all_old, pdv_1_vs_all_acctxsqa_pred)
#     counter+=1
# 
# counter=0
# fn_accfxsqa = np.empty(len(threshold_accf_x_sqa), dtype = float)
# tn_accfxsqa = np.empty(len(threshold_accf_x_sqa), dtype = float)
# fp_accfxsqa = np.empty(len(threshold_accf_x_sqa), dtype = float)
# tp_accfxsqa = np.empty(len(threshold_accf_x_sqa), dtype = float)
# fn_acctxsqa = np.empty(len(threshold_acct_x_sqa), dtype = float)
# tn_acctxsqa = np.empty(len(threshold_acct_x_sqa), dtype = float)
# fp_acctxsqa = np.empty(len(threshold_acct_x_sqa), dtype = float)
# tp_acctxsqa = np.empty(len(threshold_acct_x_sqa), dtype = float)
# 
# for i in range(0, len(threshold_accf_x_sqa)):
#     fn_accfxsqa[i] = cm_accfxsqa[i][1][0]#/len(pdv_1_vs_all)
#     tn_accfxsqa[i] = cm_accfxsqa[i][0][0]#/len(pdv_1_vs_all)
#     fp_accfxsqa[i] = cm_accfxsqa[i][0][1]#/len(pdv_1_vs_all)
#     tp_accfxsqa[i] = cm_accfxsqa[i][1][1]#/len(pdv_1_vs_all)
#     fn_acctxsqa[i] = cm_acctxsqa[i][1][0]#/len(pdv_1_vs_all)
#     tn_acctxsqa[i] = cm_acctxsqa[i][0][0]#/len(pdv_1_vs_all)
#     fp_acctxsqa[i] = cm_acctxsqa[i][0][1]#/len(pdv_1_vs_all)
#     tp_acctxsqa[i] = cm_acctxsqa[i][1][1]#/len(pdv_1_vs_all)
# 
# count_1_in_acctxsqa = np.empty(1,dtype=float)
# count_0_in_acctxsqa = np.empty(1,dtype=float)
# count_0_in_accfxsqa = np.empty(1,dtype=float)
# count_1_in_accfxsqa = np.empty(1,dtype=float)
# 
# for i in range(0, len(pdv_1_vs_all)):
#     if i==0:
#         count_1_in_accfxsqa = np.delete(count_1_in_accfxsqa, 0)
#         count_0_in_accfxsqa = np.delete(count_0_in_accfxsqa, 0)
#         count_0_in_acctxsqa = np.delete(count_0_in_acctxsqa, 0)
#         count_1_in_acctxsqa = np.delete(count_1_in_acctxsqa, 0)
#         
#     if pdv_1_vs_all[i] == 0:
#         count_0_in_accfxsqa = np.append(count_0_in_accfxsqa, i)
#         count_0_in_acctxsqa = np.append(count_0_in_acctxsqa, i)
#     if pdv_1_vs_all[i] == 1:
#         count_1_in_accfxsqa = np.append(count_1_in_accfxsqa, i)
#         count_1_in_acctxsqa = np.append(count_1_in_acctxsqa, i)
# 
# fn_accfxsqa/=len(count_1_in_accfxsqa)#/=fnsqa_accf.max()
# fp_accfxsqa/=len(count_0_in_accfxsqa)#/=fpsqa_accf.max()
# tn_accfxsqa/=len(count_0_in_accfxsqa)#/=tnsqa_accf.max()
# tp_accfxsqa/=len(count_1_in_accfxsqa)#/=tpsqa_accf.max()
# 
# fn_acctxsqa/=len(count_1_in_acctxsqa)#/=fnsqa_acct.max()
# fp_acctxsqa/=len(count_0_in_acctxsqa)#/=fpsqa_acct.max()
# tn_acctxsqa/=len(count_0_in_acctxsqa)#/=tnsqa_acct.max()
# tp_acctxsqa/=len(count_1_in_acctxsqa)#/=tpsqa_acct.max()
# =============================================================================

###################################################
###################################################
#3.ML
#initialize for ML


for i in range(0,trycount):


##a. logistic
#a1. sqa
#%%


    # clf_logit_sqa = sklm.LogisticRegression(max_iter=2000).fit(sqa_reshaped, pdv_1_vs_all)
    # #print("sqa vs pd score = ", clf_logit_sqa.score(sqa_reshaped,pdv_1_vs_all_reshaped))
    
    # cm_logit_sqa = confusion_matrix(pdv_1_vs_all, clf_logit_sqa.predict(sqa_reshaped))
    # sens_logit_sqa = cm_logit_sqa[1,1] / (cm_logit_sqa[1,1] + cm_logit_sqa[1,0])
    # spec_logit_sqa = cm_logit_sqa[0,0] / (cm_logit_sqa[0,0] + cm_logit_sqa[0,1])
    # prec_logit_sqa = cm_logit_sqa[1,1] / (cm_logit_sqa[1,1] + cm_logit_sqa[0,1])
    
    # cm_test_logit_sqa = confusion_matrix(pdv_1_vs_all_test, clf_logit_sqa.predict(sqa_test_reshaped))
    # sens_test_logit_sqa = cm_test_logit_sqa[1,1] / (cm_test_logit_sqa[1,1] + cm_test_logit_sqa[1,0])
    # spec_test_logit_sqa = cm_test_logit_sqa[0,0] / (cm_test_logit_sqa[0,0] + cm_test_logit_sqa[0,1])
    # prec_test_logit_sqa = cm_test_logit_sqa[1,1] / (cm_test_logit_sqa[1,1] + cm_test_logit_sqa[0,1])
    
    # #print("sqa logit sens, spec, prec =",sens_logit_sqa, ",",spec_logit_sqa,",", prec_logit_sqa)
    # #print(clf_logit_sqa.decision_function(sqa_reshaped))
    # #print(dict1)
    # fp_logit_sqa, tp_logit_sqa, threshold_logit_sqa = roc_curve(pdv_1_vs_all, clf_logit_sqa.predict_proba(sqa_reshaped)[:,1])
    # area_logit_sqa = auc(fp_logit_sqa,tp_logit_sqa)
    
    # logit_sqa_result= pd.DataFrame(data=[["logit sqa", sens_logit_sqa,spec_logit_sqa,prec_logit_sqa, area_logit_sqa]])
    
    # df_ML_results = pd.concat((df_ML_results, logit_sqa_result), axis=0,ignore_index=True)
    
    # # =============================================================================
    # # plt.figure(1)
    # # plt.scatter(sqa_index, clf_logit_sqa.decision_function(sqa_reshaped))
    # # plt.show()
    # # =============================================================================
    # #plt.figure(1)
    # #plt.scatter(sqa_numpy, clf_logit_sqa.predict(sqa_reshaped))
    # #plt.plot(clf_logit_sqa)
    
    
    # #a2. total_acc_frame freq
    # clf_logit_accf = sklm.LogisticRegression(max_iter=2000).fit(accf_reshaped, pdv_1_vs_all)
    # #print("accf vs pd score = ", clf_logit_accf.score(accf_reshaped,pdv_1_vs_all_reshaped))
    
    # cm_logit_accf = confusion_matrix(pdv_1_vs_all, clf_logit_accf.predict(accf_reshaped))
    # sens_logit_accf = cm_logit_accf[1,1] / (cm_logit_accf[1,1] + cm_logit_accf[1,0])
    # spec_logit_accf = cm_logit_accf[0,0] / (cm_logit_accf[0,0] + cm_logit_accf[0,1])
    # prec_logit_accf = cm_logit_accf[1,1] / (cm_logit_accf[1,1] + cm_logit_accf[0,1])
    
    # cm_test_logit_accf = confusion_matrix(pdv_1_vs_all_test, clf_logit_accf.predict(accf_test_reshaped))
    # sens_test_logit_accf = cm_test_logit_accf[1,1] / (cm_test_logit_accf[1,1] + cm_test_logit_accf[1,0])
    # spec_test_logit_accf = cm_test_logit_accf[0,0] / (cm_test_logit_accf[0,0] + cm_test_logit_accf[0,1])
    # prec_test_logit_accf = cm_test_logit_accf[1,1] / (cm_test_logit_accf[1,1] + cm_test_logit_accf[0,1])
    # #print("acc_frame_freq logit sens, spec, prec =",sens_logit_accf,",", spec_logit_accf,",", prec_logit_accf)
    # fp_logit_accf, tp_logit_accf, threshold_logit_accf = roc_curve(pdv_1_vs_all, clf_logit_accf.predict_proba(accf_reshaped)[:,1])
    # area_logit_accf = auc(fp_logit_accf,tp_logit_accf)
    
    # logit_accf_result= pd.DataFrame(data=[["logit accf", sens_logit_accf,spec_logit_accf,prec_logit_accf, area_logit_accf]]) 
    
    # df_ML_results = pd.concat((df_ML_results, logit_accf_result), ignore_index=True)
    
    
    # #a3. total_acc_frame time
    # clf_logit_acct = sklm.LogisticRegression(max_iter=2000).fit(acct_reshaped, pdv_1_vs_all)
    # #print("acct vs pd score = ", clf_logit_acct.score(acct_reshaped,pdv_1_vs_all))
    
    # cm_logit_acct = confusion_matrix(pdv_1_vs_all, clf_logit_acct.predict(acct_reshaped))
    # sens_logit_acct = cm_logit_acct[1,1] / (cm_logit_acct[1,1] + cm_logit_acct[1,0])
    # spec_logit_acct = cm_logit_acct[0,0] / (cm_logit_acct[0,0] + cm_logit_acct[0,1])
    # prec_logit_acct = cm_logit_acct[1,1] / (cm_logit_acct[1,1] + cm_logit_acct[0,1])
    
    # cm_test_logit_acct = confusion_matrix(pdv_1_vs_all_test, clf_logit_acct.predict(acct_test_reshaped))
    # sens_test_logit_acct = cm_test_logit_acct[1,1] / (cm_test_logit_acct[1,1] + cm_test_logit_acct[1,0])
    # spec_test_logit_acct = cm_test_logit_acct[0,0] / (cm_test_logit_acct[0,0] + cm_test_logit_acct[0,1])
    # prec_test_logit_acct = cm_test_logit_acct[1,1] / (cm_test_logit_acct[1,1] + cm_test_logit_acct[0,1])
    # #print("acc_frame_time logit sens, spec, prec =",sens_logit_acct,",", spec_logit_acct,",", prec_logit_acct)
    # fp_logit_acct, tp_logit_acct, threshold_logit_acct = roc_curve(pdv_1_vs_all, clf_logit_acct.predict_proba(acct_reshaped)[:,1])
    # area_logit_acct = auc(fp_logit_acct,tp_logit_acct)
    
    # logit_acct_result= pd.DataFrame(data=[["logit acct", sens_logit_acct,spec_logit_acct,prec_logit_acct, area_logit_acct]]) 
    
    # df_ML_results = pd.concat((df_ML_results, logit_acct_result), ignore_index=True)
    
    
    
    # #a4. total_acc_frame freq x SQA
    # clf_logit_accf_x_sqa = sklm.LogisticRegression(max_iter=2000).fit(df_2, pdv_1_vs_all)
    # #print("accf_x_sqa vs pd score = ", clf_logit_accf_x_sqa.score(df_2,pdv_1_vs_all))
    
    # cm_logit_accf_x_sqa = confusion_matrix(pdv_1_vs_all, clf_logit_accf_x_sqa.predict(df_2))
    # sens_logit_accf_x_sqa = cm_logit_accf_x_sqa[1,1] / (cm_logit_accf_x_sqa[1,1] + cm_logit_accf_x_sqa[1,0])
    # spec_logit_accf_x_sqa = cm_logit_accf_x_sqa[0,0] / (cm_logit_accf_x_sqa[0,0] + cm_logit_accf_x_sqa[0,1])
    # prec_logit_accf_x_sqa = cm_logit_accf_x_sqa[1,1] / (cm_logit_accf_x_sqa[1,1] + cm_logit_accf_x_sqa[0,1])
    
    # cm_test_logit_accf_x_sqa = confusion_matrix(pdv_1_vs_all_test, clf_logit_accf_x_sqa.predict(df_test_2))
    # sens_test_logit_accf_x_sqa = cm_test_logit_accf_x_sqa[1,1] / (cm_test_logit_accf_x_sqa[1,1] + cm_test_logit_accf_x_sqa[1,0])
    # spec_test_logit_accf_x_sqa = cm_test_logit_accf_x_sqa[0,0] / (cm_test_logit_accf_x_sqa[0,0] + cm_test_logit_accf_x_sqa[0,1])
    # prec_test_logit_accf_x_sqa = cm_test_logit_accf_x_sqa[1,1] / (cm_test_logit_accf_x_sqa[1,1] + cm_test_logit_accf_x_sqa[0,1])
    # #print("acc_frame_time logit sens, spec, prec =",sens_logit_accf_x_sqa,",", spec_logit_accf_x_sqa,",", prec_logit_accf_x_sqa)
    # fp_logit_accf_x_sqa, tp_logit_accf_x_sqa, threshold_logit_accf_x_sqa = roc_curve(pdv_1_vs_all, clf_logit_accf_x_sqa.predict_proba(df_2)[:,1])
    # area_logit_accf_x_sqa = auc(fp_logit_accf_x_sqa,tp_logit_accf_x_sqa)
    
    # logit_accf_x_sqa_result= pd.DataFrame(data=[["logit accf_x_sqa", sens_logit_accf_x_sqa,spec_logit_accf_x_sqa,prec_logit_accf_x_sqa, area_logit_accf_x_sqa] ])
    
    # df_ML_results = pd.concat((df_ML_results, logit_accf_x_sqa_result), ignore_index=True)
    
    #   #a5. total_acc_frame time x SQA
    # clf_logit_acct_x_sqa = sklm.LogisticRegression(max_iter=2000).fit(df_3, pdv_1_vs_all)
    # #print("acct_x_sqa vs pd score = ", clf_logit_acct_x_sqa.score(df_3,pdv_1_vs_all))
    
    # cm_logit_acct_x_sqa = confusion_matrix(pdv_1_vs_all, clf_logit_acct_x_sqa.predict(df_3))
    # sens_logit_acct_x_sqa = cm_logit_acct_x_sqa[1,1] / (cm_logit_acct_x_sqa[1,1] + cm_logit_acct_x_sqa[1,0])
    # spec_logit_acct_x_sqa = cm_logit_acct_x_sqa[0,0] / (cm_logit_acct_x_sqa[0,0] + cm_logit_acct_x_sqa[0,1])
    # prec_logit_acct_x_sqa = cm_logit_acct_x_sqa[1,1] / (cm_logit_acct_x_sqa[1,1] + cm_logit_acct_x_sqa[0,1])
    
    # cm_test_logit_acct_x_sqa = confusion_matrix(pdv_1_vs_all_test, clf_logit_acct_x_sqa.predict(df_test_3))
    # sens_test_logit_acct_x_sqa = cm_test_logit_acct_x_sqa[1,1] / (cm_test_logit_acct_x_sqa[1,1] + cm_test_logit_acct_x_sqa[1,0])
    # spec_test_logit_acct_x_sqa = cm_test_logit_acct_x_sqa[0,0] / (cm_test_logit_acct_x_sqa[0,0] + cm_test_logit_acct_x_sqa[0,1])
    # prec_test_logit_acct_x_sqa = cm_test_logit_acct_x_sqa[1,1] / (cm_test_logit_acct_x_sqa[1,1] + cm_test_logit_acct_x_sqa[0,1])
    # #print("acc_frame_time logit sens, spec, prec =",sens_logit_acct_x_sqa,",", spec_logit_acct_x_sqa,",", prec_logit_acct_x_sqa)
    # fp_logit_acct_x_sqa, tp_logit_acct_x_sqa, threshold_logit_acct_x_sqa = roc_curve(pdv_1_vs_all, clf_logit_acct_x_sqa.predict_proba(df_3)[:,1])
    # area_logit_acct_x_sqa = auc(fp_logit_acct_x_sqa,tp_logit_acct_x_sqa)
    
    # logit_acct_x_sqa_result= pd.DataFrame(data=[["logit acct_x_sqa", sens_logit_acct_x_sqa,spec_logit_acct_x_sqa,prec_logit_acct_x_sqa, area_logit_acct_x_sqa]]) 
    
    # df_ML_results = pd.concat((df_ML_results, logit_acct_x_sqa_result), ignore_index=True)
    
    # #a6. total_acc_frame freq, time x SQA
    # clf_logit_accft_x_sqa = sklm.LogisticRegression(max_iter=2000).fit(df_predictor_1, pdv_1_vs_all)
    # #print("accft_x_sqa vs pd score = ", clf_logit_accft_x_sqa.score(df_predictor_1,pdv_1_vs_all))
    
    # cm_logit_accft_x_sqa = confusion_matrix(pdv_1_vs_all, clf_logit_accft_x_sqa.predict(df_predictor_1))
    # sens_logit_accft_x_sqa = cm_logit_accft_x_sqa[1,1] / (cm_logit_accft_x_sqa[1,1] + cm_logit_accft_x_sqa[1,0])
    # spec_logit_accft_x_sqa = cm_logit_accft_x_sqa[0,0] / (cm_logit_accft_x_sqa[0,0] + cm_logit_accft_x_sqa[0,1])
    # prec_logit_accft_x_sqa = cm_logit_accft_x_sqa[1,1] / (cm_logit_accft_x_sqa[1,1] + cm_logit_accft_x_sqa[0,1])
    
    # cm_test_logit_accft_x_sqa = confusion_matrix(pdv_1_vs_all_test, clf_logit_accft_x_sqa.predict(df_test_predictor_1))
    # sens_test_logit_accft_x_sqa = cm_test_logit_accft_x_sqa[1,1] / (cm_test_logit_accft_x_sqa[1,1] + cm_test_logit_accft_x_sqa[1,0])
    # spec_test_logit_accft_x_sqa = cm_test_logit_accft_x_sqa[0,0] / (cm_test_logit_accft_x_sqa[0,0] + cm_test_logit_accft_x_sqa[0,1])
    # prec_test_logit_accft_x_sqa = cm_test_logit_accft_x_sqa[1,1] / (cm_test_logit_accft_x_sqa[1,1] + cm_test_logit_accft_x_sqa[0,1])
    # #print("acc_frame_time logit sens, spec, prec =",sens_logit_accft_x_sqa,",", spec_logit_accft_x_sqa,",", prec_logit_accft_x_sqa)
    # fp_logit_accft_x_sqa, tp_logit_accft_x_sqa, threshold_logit_accft_x_sqa = roc_curve(pdv_1_vs_all, clf_logit_accft_x_sqa.predict_proba(df_predictor_1)[:,1])
    # area_logit_accft_x_sqa = auc(fp_logit_accft_x_sqa,tp_logit_accft_x_sqa)
    
    # logit_accft_x_sqa_result= pd.DataFrame(data=[["logit accft_x_sqa", sens_logit_accft_x_sqa,spec_logit_accft_x_sqa,prec_logit_accft_x_sqa, area_logit_accft_x_sqa]]) 
    
    # df_ML_results = pd.concat((df_ML_results, logit_accft_x_sqa_result), ignore_index=True)
    
    
    
    # #a7. all logistic all logit
    clf_logit_all = sklm.LogisticRegression(C=2.0,penalty='l1', solver='saga', max_iter=3000, tol=1e-3).fit(df_4, pdv_1_vs_all)
    #print("all vs pd score = ", clf_logit_all.score(df_4,pdv_1_vs_all))
    
    pred_nd_logit = clf_logit_all.predict(df_4_nodrop)
    df_diff_logit = pd.DataFrame()
    loopcounter=0
    for i in pdvnd_1_vs_all.index:
        if ((pdvnd_1_vs_all[i]!=1) and (pred_nd_logit[loopcounter]==1)):
            df_diff_logit = pd.concat((df_diff_logit, pd.Series([i])))
        loopcounter+=1
            
    pred_nd_logit_test = clf_logit_all.predict(df_test_4_nodrop)     
    loopcounter=0  
    df_diff_logit = pd.DataFrame()
    for i in pdvnd_1_vs_all_test.index:
        if ((pdvnd_1_vs_all_test[i]!=1) and (pred_nd_logit_test[loopcounter]==1)):
            df_diff_logit = pd.concat((df_diff_logit, pd.Series([i])))
        loopcounter+=1
    
    cm_logit_all = confusion_matrix(pdv_1_vs_all, clf_logit_all.predict(df_4))
    sens_logit_all = cm_logit_all[1,1] / (cm_logit_all[1,1] + cm_logit_all[1,0])
    spec_logit_all = cm_logit_all[0,0] / (cm_logit_all[0,0] + cm_logit_all[0,1])
    prec_logit_all = cm_logit_all[1,1] / (cm_logit_all[1,1] + cm_logit_all[0,1])
    
    cm_logit_all_nodrop = confusion_matrix(pdvnd_1_vs_all, clf_logit_all.predict(df_4_nodrop))
    sens_logit_all_nodrop = cm_logit_all_nodrop[1,1] / (cm_logit_all_nodrop[1,1] + cm_logit_all_nodrop[1,0])
    spec_logit_all_nodrop = cm_logit_all_nodrop[0,0] / (cm_logit_all_nodrop[0,0] + cm_logit_all_nodrop[0,1])
    prec_logit_all_nodrop = cm_logit_all_nodrop[1,1] / (cm_logit_all_nodrop[1,1] + cm_logit_all_nodrop[0,1])
    
    cm_test_logit_all = confusion_matrix(pdv_1_vs_all_test, clf_logit_all.predict(df_test_4))
    sens_test_logit_all = cm_test_logit_all[1,1] / (cm_test_logit_all[1,1] + cm_test_logit_all[1,0])
    spec_test_logit_all = cm_test_logit_all[0,0] / (cm_test_logit_all[0,0] + cm_test_logit_all[0,1])
    prec_test_logit_all = cm_test_logit_all[1,1] / (cm_test_logit_all[1,1] + cm_test_logit_all[0,1])
    
    cm_test_logit_all_nodrop = confusion_matrix(pdvnd_1_vs_all_test, clf_logit_all.predict(df_test_4_nodrop))
    sens_test_logit_all_nodrop = cm_test_logit_all_nodrop[1,1] / (cm_test_logit_all_nodrop[1,1] + cm_test_logit_all_nodrop[1,0])
    spec_test_logit_all_nodrop = cm_test_logit_all_nodrop[0,0] / (cm_test_logit_all_nodrop[0,0] + cm_test_logit_all_nodrop[0,1])
    prec_test_logit_all_nodrop = cm_test_logit_all_nodrop[1,1] / (cm_test_logit_all_nodrop[1,1] + cm_test_logit_all_nodrop[0,1])
    
    # ccc=0
    # for k in pdvnd_1_vs_all.index:
    #     if (pdvnd_1_vs_all[pdvnd_1_vs_all.index[ccc]]==1 and clf_logit_all.predict(df_4_nodrop)[ccc]==0):
    #         df_fn_results = pd.concat((df_fn_results, df_nodrop[df_nodrop.index[ccc]].astype('Series')))
    #     if (pdvnd_1_vs_all[pdvnd_1_vs_all.index[ccc]]==0 and clf_logit_all.predict(df_4_nodrop)[ccc]==1):
    #         df_fp_results = pd.concat((df_fp_results, df_nodrop[df_nodrop.index[ccc]].astype('Series')))
    #     ccc+=1
    xyxyx = clf_logit_all.predict(df_4_nodrop)    
    # ccc=0
    # for k in pdvnd_1_vs_all.index:
    #     if (pdvnd_1_vs_all[k]==1 and clf_logit_all.predict(df_4_nodrop)[ccc]==0):
    #         df_fn_results = pd.concat((df_fn_results, df_nodrop[k].astype('Series')))
    #     if (pdvnd_1_vs_all[k]==0 and clf_logit_all.predict(df_4_nodrop)[ccc]==1):
    #         df_fp_results = pd.concat((df_fp_results, df_nodrop[k].astype('Series')))
    #     ccc+=1    
    
    
    #print("acc_frame_time logit sens, spec, prec =",sens_logit_all,",", spec_logit_all,",", prec_logit_all)
    fp_logit_all, tp_logit_all, threshold_logit_all = roc_curve(pdv_1_vs_all, clf_logit_all.predict_proba(df_4)[:,1])
    area_logit_all = auc(fp_logit_all,tp_logit_all)
    
    logit_all_result= pd.DataFrame(data=[["logit all", sens_logit_all,spec_logit_all,prec_logit_all, area_logit_all]])
    
    df_ML_results = pd.concat((df_ML_results, logit_all_result), ignore_index=True)

    print(clf_logit_all.coef_)
    plt.figure(11)
    perm_importance_logit = permutation_importance(clf_logit_all, df_test_4, pdv_1_vs_all_test)

    sorted_idx = perm_importance_logit.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance_logit.importances_mean[sorted_idx])
    plt.xlabel("Feature Importance (Logit)")
    plt.show()
    
    
    ##b. Neural Network
    #%%
    # clf_nn_sqa = MLPClassifier(activation='tanh',hidden_layer_sizes=(10,6), max_iter=2000, alpha=0, learning_rate_init=0.002).fit(sqa_reshaped, pdv_1_vs_all)
    # cm_nn_sqa = confusion_matrix(pdv_1_vs_all, clf_nn_sqa.predict(sqa_reshaped))
    # sens_nn_sqa = cm_nn_sqa[1,1] / (cm_nn_sqa[1,1] + cm_nn_sqa[1,0])
    # spec_nn_sqa = cm_nn_sqa[0,0] / (cm_nn_sqa[0,0] + cm_nn_sqa[0,1])
    # prec_nn_sqa = cm_nn_sqa[1,1] / (cm_nn_sqa[1,1] + cm_nn_sqa[0,1])
    
    # cm_test_nn_sqa = confusion_matrix(pdv_1_vs_all_test, clf_nn_sqa.predict(sqa_test_reshaped))
    # sens_test_nn_sqa = cm_test_nn_sqa[1,1] / (cm_test_nn_sqa[1,1] + cm_test_nn_sqa[1,0])
    # spec_test_nn_sqa = cm_test_nn_sqa[0,0] / (cm_test_nn_sqa[0,0] + cm_test_nn_sqa[0,1])
    # prec_test_nn_sqa = cm_test_nn_sqa[1,1] / (cm_test_nn_sqa[1,1] + cm_test_nn_sqa[0,1])
    
    # # cm_test_logit_sqa = confusion_matrix(pdv_1_vs_all_test, clf_logit_sqa.predict(sqa_test_reshaped))
    # # sens_test_logit_sqa = cm_test_logit_sqa[1,1] / (cm_test_logit_sqa[1,1] + cm_test_logit_sqa[1,0])
    # # spec_test_logit_sqa = cm_test_logit_sqa[0,0] / (cm_test_logit_sqa[0,0] + cm_test_logit_sqa[0,1])
    # # prec_test_logit_sqa = cm_test_logit_sqa[1,1] / (cm_test_logit_sqa[1,1] + cm_test_logit_sqa[0,1])
    # #print("sqa neural net sens, spec, prec =",sens_nn_sqa,",", spec_nn_sqa,",", prec_nn_sqa)
    
    # fp_sqa, tp_sqa, threshold_nn_sqa = roc_curve(pdv_1_vs_all, clf_nn_sqa.predict_proba(sqa_reshaped)[:,1])
    # area_nn_sqa = auc(fp_sqa,tp_sqa)
    
    # nn_sqa_result= pd.DataFrame(data=[["nn sqa", sens_nn_sqa,spec_nn_sqa,prec_nn_sqa, area_nn_sqa]])
    
    # df_ML_results = pd.concat((df_ML_results, nn_sqa_result), ignore_index=True)
    
    
    # clf_nn_accf = MLPClassifier(activation='tanh',hidden_layer_sizes=(10,6), max_iter=2000, alpha=0, learning_rate_init=0.005).fit(accf_reshaped, pdv_1_vs_all)
    # cm_nn_accf = confusion_matrix(pdv_1_vs_all, clf_nn_accf.predict(accf_reshaped))
    # sens_nn_accf = cm_nn_accf[1,1] / (cm_nn_accf[1,1] + cm_nn_accf[1,0])
    # spec_nn_accf = cm_nn_accf[0,0] / (cm_nn_accf[0,0] + cm_nn_accf[0,1])
    # prec_nn_accf = cm_nn_accf[1,1] / (cm_nn_accf[1,1] + cm_nn_accf[0,1])
    
    # cm_test_nn_accf = confusion_matrix(pdv_1_vs_all_test, clf_nn_accf.predict(accf_test_reshaped))
    # sens_test_nn_accf = cm_test_nn_accf[1,1] / (cm_test_nn_accf[1,1] + cm_test_nn_accf[1,0])
    # spec_test_nn_accf = cm_test_nn_accf[0,0] / (cm_test_nn_accf[0,0] + cm_test_nn_accf[0,1])
    # prec_test_nn_accf = cm_test_nn_accf[1,1] / (cm_test_nn_accf[1,1] + cm_test_nn_accf[0,1])
    
    # # cm_test_logit_accf = confusion_matrix(pdv_1_vs_all_test, clf_logit_accf.predict(accf_test_reshaped))
    # # sens_test_logit_accf = cm_test_logit_accf[1,1] / (cm_test_logit_accf[1,1] + cm_test_logit_accf[1,0])
    # # spec_test_logit_accf = cm_test_logit_accf[0,0] / (cm_test_logit_accf[0,0] + cm_test_logit_accf[0,1])
    # # prec_test_logit_accf = cm_test_logit_accf[1,1] / (cm_test_logit_accf[1,1] + cm_test_logit_accf[0,1])
    # #print("accf neural net sens, spec, prec =",sens_nn_accf,",", spec_nn_accf,",", prec_nn_accf)
    
    # fp_accf, tp_accf, threshold_nn_accf = roc_curve(pdv_1_vs_all, clf_nn_accf.predict_proba(accf_reshaped)[:,1])
    # area_nn_accf = auc(fp_accf,tp_accf)
    
    # nn_accf_result= pd.DataFrame(data=[["nn accf", sens_nn_accf,spec_nn_accf,prec_nn_accf, area_nn_accf]])
    
    # df_ML_results = pd.concat((df_ML_results, nn_accf_result), ignore_index=True)
    
    
    # clf_nn_acct = MLPClassifier(activation='tanh',hidden_layer_sizes=(10,6), max_iter=2000, alpha=0, learning_rate_init=0.005).fit(acct_reshaped, pdv_1_vs_all)
    # cm_nn_acct = confusion_matrix(pdv_1_vs_all, clf_nn_acct.predict(acct_reshaped))
    # sens_nn_acct = cm_nn_acct[1,1] / (cm_nn_acct[1,1] + cm_nn_acct[1,0])
    # spec_nn_acct = cm_nn_acct[0,0] / (cm_nn_acct[0,0] + cm_nn_acct[0,1])
    # prec_nn_acct = cm_nn_acct[1,1] / (cm_nn_acct[1,1] + cm_nn_acct[0,1])
    
    # cm_test_nn_acct = confusion_matrix(pdv_1_vs_all_test, clf_nn_acct.predict(acct_test_reshaped))
    # sens_test_nn_acct = cm_test_nn_acct[1,1] / (cm_test_nn_acct[1,1] + cm_test_nn_acct[1,0])
    # spec_test_nn_acct = cm_test_nn_acct[0,0] / (cm_test_nn_acct[0,0] + cm_test_nn_acct[0,1])
    # prec_test_nn_acct = cm_test_nn_acct[1,1] / (cm_test_nn_acct[1,1] + cm_test_nn_acct[0,1])
    # #print("acct neural net sens, spec, prec =",sens_nn_acct,",", spec_nn_acct,",", prec_nn_acct)
    
    # fp_acct, tp_acct, threshold_nn_acct = roc_curve(pdv_1_vs_all, clf_nn_acct.predict_proba(acct_reshaped)[:,1])
    # area_nn_acct = auc(fp_acct,tp_acct)
    
    # nn_acct_result= pd.DataFrame(data=[["nn acct", sens_nn_acct,spec_nn_acct,prec_nn_acct, area_nn_acct]])
    
    # df_ML_results = pd.concat((df_ML_results, nn_acct_result), ignore_index=True)
    
    
    # #%%
    # clf_nn_accf_x_sqa = MLPClassifier(activation='tanh',hidden_layer_sizes=(10,8,6), max_iter=2000, alpha=0, learning_rate_init=0.004).fit(df_2, pdv_1_vs_all)
    # cm_nn_accf_x_sqa = confusion_matrix(pdv_1_vs_all, clf_nn_accf_x_sqa.predict(df_2))
    # sens_nn_accf_x_sqa = cm_nn_accf_x_sqa[1,1] / (cm_nn_accf_x_sqa[1,1] + cm_nn_accf_x_sqa[1,0])
    # spec_nn_accf_x_sqa = cm_nn_accf_x_sqa[0,0] / (cm_nn_accf_x_sqa[0,0] + cm_nn_accf_x_sqa[0,1])
    # prec_nn_accf_x_sqa = cm_nn_accf_x_sqa[1,1] / (cm_nn_accf_x_sqa[1,1] + cm_nn_accf_x_sqa[0,1])
    
    # cm_test_nn_accf_x_sqa = confusion_matrix(pdv_1_vs_all_test, clf_nn_accf_x_sqa.predict(df_test_2))
    # sens_test_nn_accf_x_sqa = cm_test_nn_accf_x_sqa[1,1] / (cm_test_nn_accf_x_sqa[1,1] + cm_test_nn_accf_x_sqa[1,0])
    # spec_test_nn_accf_x_sqa = cm_test_nn_accf_x_sqa[0,0] / (cm_test_nn_accf_x_sqa[0,0] + cm_test_nn_accf_x_sqa[0,1])
    # prec_test_nn_accf_x_sqa = cm_test_nn_accf_x_sqa[1,1] / (cm_test_nn_accf_x_sqa[1,1] + cm_test_nn_accf_x_sqa[0,1])
    # #print("accf_x_sqa neural net sens, spec, prec =",sens_nn_accf_x_sqa,",", spec_nn_accf_x_sqa,",", prec_nn_accf_x_sqa)
    
    # fp_accf_x_sqa, tp_accf_x_sqa, threshold_nn_accf_x_sqa = roc_curve(pdv_1_vs_all, clf_nn_accf_x_sqa.predict_proba(df_2)[:,1])
    # area_nn_accf_x_sqa = auc(fp_accf_x_sqa,tp_accf_x_sqa)
    
    # nn_accf_x_sqa_result= pd.DataFrame(data=[["nn accf_x_sqa", sens_nn_accf_x_sqa,spec_nn_accf_x_sqa,prec_nn_accf_x_sqa, area_nn_accf_x_sqa]]) 
    
    # df_ML_results = pd.concat((df_ML_results, nn_accf_x_sqa_result), ignore_index=True)
    
    
    # clf_nn_acct_x_sqa = MLPClassifier(activation='tanh',hidden_layer_sizes=(10,8,6), max_iter=2000, alpha=0, learning_rate_init=0.01).fit(df_3, pdv_1_vs_all)
    # cm_nn_acct_x_sqa = confusion_matrix(pdv_1_vs_all, clf_nn_acct_x_sqa.predict(df_3))
    # sens_nn_acct_x_sqa = cm_nn_acct_x_sqa[1,1] / (cm_nn_acct_x_sqa[1,1] + cm_nn_acct_x_sqa[1,0])
    # spec_nn_acct_x_sqa = cm_nn_acct_x_sqa[0,0] / (cm_nn_acct_x_sqa[0,0] + cm_nn_acct_x_sqa[0,1])
    # prec_nn_acct_x_sqa = cm_nn_acct_x_sqa[1,1] / (cm_nn_acct_x_sqa[1,1] + cm_nn_acct_x_sqa[0,1])
    
    # cm_test_nn_acct_x_sqa = confusion_matrix(pdv_1_vs_all_test, clf_nn_acct_x_sqa.predict(df_test_3))
    # sens_test_nn_acct_x_sqa = cm_test_nn_acct_x_sqa[1,1] / (cm_test_nn_acct_x_sqa[1,1] + cm_test_nn_acct_x_sqa[1,0])
    # spec_test_nn_acct_x_sqa = cm_test_nn_acct_x_sqa[0,0] / (cm_test_nn_acct_x_sqa[0,0] + cm_test_nn_acct_x_sqa[0,1])
    # prec_test_nn_acct_x_sqa = cm_test_nn_acct_x_sqa[1,1] / (cm_test_nn_acct_x_sqa[1,1] + cm_test_nn_acct_x_sqa[0,1])
    # #print("acct_x_sqa neural net sens, spec, prec =",sens_nn_acct_x_sqa,",", spec_nn_acct_x_sqa,",", prec_nn_acct_x_sqa)
    
    # fp_acct_x_sqa, tp_acct_x_sqa, threshold_nn_acct_x_sqa = roc_curve(pdv_1_vs_all, clf_nn_acct_x_sqa.predict_proba(df_3)[:,1])
    # area_nn_acct_x_sqa = auc(fp_acct_x_sqa,tp_acct_x_sqa)
    
    # nn_acct_x_sqa_result= pd.DataFrame(data=[["nn acct_x_sqa", sens_nn_acct_x_sqa,spec_nn_acct_x_sqa,prec_nn_acct_x_sqa, area_nn_acct_x_sqa]]) 
    
    # df_ML_results = pd.concat((df_ML_results, nn_acct_x_sqa_result), ignore_index=True)
    
    # #print(auc(fp_acct_x_sqa, tp_acct_x_sqa))
    
    
    
    # clf_nn_accft_x_sqa = MLPClassifier(activation='tanh',hidden_layer_sizes=(40,10,5), max_iter=2000, learning_rate_init=0.001, alpha=0.0001).fit(df_predictor_1, pdv_1_vs_all)

    # cm_nn_accft_x_sqa = confusion_matrix(pdv_1_vs_all, clf_nn_accft_x_sqa.predict(df_predictor_1))
    
    # sens_nn_accft_x_sqa = cm_nn_accft_x_sqa[1,1] / (cm_nn_accft_x_sqa[1,1] + cm_nn_accft_x_sqa[1,0])
    # spec_nn_accft_x_sqa = cm_nn_accft_x_sqa[0,0] / (cm_nn_accft_x_sqa[0,0] + cm_nn_accft_x_sqa[0,1])
    # prec_nn_accft_x_sqa = cm_nn_accft_x_sqa[1,1] / (cm_nn_accft_x_sqa[1,1] + cm_nn_accft_x_sqa[0,1])
    
    # cm_test_nn_accft_x_sqa = confusion_matrix(pdv_1_vs_all_test, clf_nn_accft_x_sqa.predict(df_test_predictor_1))
    # sens_test_nn_accft_x_sqa = cm_test_nn_accft_x_sqa[1,1] / (cm_test_nn_accft_x_sqa[1,1] + cm_test_nn_accft_x_sqa[1,0])
    # spec_test_nn_accft_x_sqa = cm_test_nn_accft_x_sqa[0,0] / (cm_test_nn_accft_x_sqa[0,0] + cm_test_nn_accft_x_sqa[0,1])
    # prec_test_nn_accft_x_sqa = cm_test_nn_accft_x_sqa[1,1] / (cm_test_nn_accft_x_sqa[1,1] + cm_test_nn_accft_x_sqa[0,1])
    
    # #print("accft_x_sqa neural net sens, spec, prec =",sens_nn_accft_x_sqa,",", spec_nn_accft_x_sqa,",", prec_nn_accft_x_sqa)
    # fp_accft_x_sqa, tp_accft_x_sqa, threshold_nn_accft_x_sqa = roc_curve(pdv_1_vs_all, clf_nn_accft_x_sqa.predict_proba(df_predictor_1)[:,1])
    # area_nn_accft_x_sqa = auc(fp_accft_x_sqa,tp_accft_x_sqa)
    
    # nn_accft_x_sqa_result= pd.DataFrame(data=[["nn accft_x_sqa", sens_nn_accft_x_sqa,spec_nn_accft_x_sqa,prec_nn_accft_x_sqa, area_nn_accft_x_sqa]]) 
    
    # df_ML_results = pd.concat((df_ML_results, nn_accft_x_sqa_result), ignore_index=True)
    # #print(auc(fp_accft_x_sqa, tp_accft_x_sqa))
    



    #hyperparameters optimization for all (12) features
    # clf_nn_all = MLPClassifier()
    # cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)
    # space=dict()
    # space['solver'] = ['adam']
    # space['alpha'] = np.logspace(0.0001,10,12)
    # space['learning_rate_init'] = np.logspace(0.0001, 1, 10)
    # space['max_iter']=[2000]
    # space['hidden_layer_sizes']=[(128), (256), (512), (128, 64), (256,128), (512,256), (256,128, 64)]
    # space['activation'] = ['relu']
    # # space['solver'] = ['adam']
    # # space['alpha'] = np.logspace(0.0001,10,2)
    # # space['learning_rate_init'] = np.logspace(0.0001, 1, 2)
    # # space['max_iter']=[2000]
    # # space['hidden_layer_sizes']=[(128)]
    # # space['activation'] = ['relu']
    # search = GridSearchCV(clf_nn_all, space, scoring='roc_auc', n_jobs=-1, cv=cv, verbose=3)
    
    # print("now fitting...")
    # result = search.fit(df_4, pdv_1_vs_all)
    # # summarize result
    # print('Best Score: %s' % result.best_score_)
    # print('Best Hyperparameters: %s' % result.best_params_)
    # print('RESULTS: %s' % result.cv_results_)
      
    #temporary comment to nn all, for hyperparameters optimization gridsearch 
    clf_nn_all = MLPClassifier(activation='relu',hidden_layer_sizes=(1000,300), max_iter=6000, alpha=0.001, learning_rate_init=0.001, solver='adam', tol=1e-4).fit(df_4, pdv_1_vs_all)
    #,early_stopping=True
    #clf_nn_all = SelectFromModel(clf_nn_all,threshold="0.5*mean",prefit=False, importance_getter='roc_auc').fit_transform(df_4, pdv_1_vs_all)#, threshold="200*mean", max_features=10, importance_getter='roc_auc', prefit=True
    
    pred_nd_nn = clf_nn_all.predict(df_4_nodrop).copy()
    df_diff_nn = pd.DataFrame()
    # loopcounter=0
    # for i in pdvnd_1_vs_all.index:
    #     if ((pdvnd_1_vs_all[i]!=1) and (pred_nd_nn[loopcounter]==1)):
    #         df_diff_nn = pd.concat((df_diff_nn, pd.Series([i])))
    #     loopcounter+=1
            
    # pred_nd_nn_test = clf_nn_all.predict(df_test_4_nodrop)     
    # loopcounter=0  
    # df_diff_nn = pd.DataFrame()
    # for i in pdvnd_1_vs_all_test.index:
    #     if ((pdvnd_1_vs_all_test[i]!=1) and (pred_nd_nn_test[loopcounter]==1)):
    #         df_diff_nn = pd.concat((df_diff_nn, pd.Series([i])))
    #     loopcounter+=1
    loopcounter=0
    for i in pdvnd_1_vs_all.index:
        if ((pdvnd_1_vs_all[i]==1) and (pred_nd_nn[loopcounter]!=1)):
            df_diff_nn = pd.concat((df_diff_nn, pd.Series([i])))
        loopcounter+=1
            
    pred_nd_nn_test = clf_nn_all.predict(df_test_4_nodrop)     
    loopcounter=0  
    df_diff_nn = pd.DataFrame()
    for i in pdvnd_1_vs_all_test.index:
        if ((pdvnd_1_vs_all_test[i]==1) and (pred_nd_nn_test[loopcounter]!=1)):
            df_diff_nn = pd.concat((df_diff_nn, pd.Series([i])))
        loopcounter+=1
        
    
    cm_nn_all = confusion_matrix(pdv_1_vs_all, clf_nn_all.predict(df_4))
    cm_test_nn_all = confusion_matrix(pdv_1_vs_all_test, clf_nn_all.predict(df_test_4))
    
    
    sens_nn_all = cm_nn_all[1,1] / (cm_nn_all[1,1] + cm_nn_all[1,0])
    spec_nn_all = cm_nn_all[0,0] / (cm_nn_all[0,0] + cm_nn_all[0,1])
    prec_nn_all = cm_nn_all[1,1] / (cm_nn_all[1,1] + cm_nn_all[0,1])
    sens_test_nn_all = cm_test_nn_all[1,1] / (cm_test_nn_all[1,1] + cm_test_nn_all[1,0])
    spec_test_nn_all = cm_test_nn_all[0,0] / (cm_test_nn_all[0,0] + cm_test_nn_all[0,1])
    prec_test_nn_all = cm_test_nn_all[1,1] / (cm_test_nn_all[1,1] + cm_test_nn_all[0,1])
    #print("all neural net sens, spec, prec =",sens_nn_all,",", spec_nn_all,",", prec_nn_all)
    
    fp_all, tp_all, threshold_nn_all = roc_curve(pdv_1_vs_all, clf_nn_all.predict_proba(df_4)[:,1])
    area_nn_all = auc(fp_all,tp_all)
    
    nn_all_result= pd.DataFrame(data=[["nn all", sens_nn_all,spec_nn_all,prec_nn_all, area_nn_all]])
    
    df_ML_results = pd.concat((df_ML_results, nn_all_result), ignore_index=True)
    
    plt.figure(9)
    perm_importance_nn = permutation_importance(clf_nn_all, df_test_4, pdv_1_vs_all_test)

    sorted_idx = perm_importance_nn.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance_nn.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance (NN)")
    plt.show()
    
    ##naive bayes
    # import sklearn.naive_bayes as nby
    # gnb = nby.GaussianNB()
    # cnb = nby.ComplementNB(norm=False,alpha=1.0)
    # gnb_all_pred = gnb.fit(df_4, pdv_1_vs_all).predict(df_test_4)
    # cnb_all_pred = cnb.fit(df_4, pdv_1_vs_all).predict(df_test_4)
    # print("naive bayes accuracy=%lf, %lf"%(1-(pdv_1_vs_all_test != gnb_all_pred).sum()/df_test_4.shape[0], 1-(pdv_1_vs_all_test != cnb_all_pred).sum()/df_test_4.shape[0]))
    
    
    ##############
    #4. visualization, results
    ##a. plots
    # plt.figure(1)
    # # plt.plot(fp_sqa,tp_sqa, 'm-', label ='sqa features, AUC = %.3f'%area_nn_sqa)
    # # plt.plot(fp_accf,tp_accf, 'c-', label ='accf features, AUC = %.3f'%area_nn_accf)
    # # plt.plot(fp_acct,tp_acct, 'k-', label ='acct features, AUC = %.3f'%area_nn_acct)
    # # plt.plot(fp_accf_x_sqa,tp_accf_x_sqa, 'y-', label ='accf_x_sqa features, AUC = %.3f'%area_nn_accf_x_sqa)
    # # plt.plot(fp_acct_x_sqa,tp_acct_x_sqa, 'b-', label ='acct_x_sqa features, AUC = %.3f'%area_nn_acct_x_sqa)
    # # plt.plot(fp_accft_x_sqa,tp_accft_x_sqa, 'g-', label ='accft_x_sqa features, AUC = %.3f'%area_nn_accft_x_sqa)
    # plt.plot(fp_all,tp_all, 'r-', label ='all features, AUC = %.3f'%area_nn_all)
    # #print(auc(fp_all, tp_all))
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC of features using NN (Training)')
    # plt.legend()
    # plt.show()
    
    
    
    # plt.figure(2)
    # # plt.plot(fp_logit_sqa,tp_logit_sqa, 'm-', label ='sqa features, AUC = %.3f'%area_logit_sqa)
    # # plt.plot(fp_logit_accf,tp_logit_accf, 'c-', label ='accf features, AUC = %.3f'%area_logit_accf)
    # # plt.plot(fp_logit_acct,tp_logit_acct, 'k-', label ='acct features, AUC = %.3f'%area_logit_acct)
    # # plt.plot(fp_logit_accf_x_sqa,tp_logit_accf_x_sqa, 'y-', label ='accf_x_sqa features, AUC = %.3f'%area_logit_accf_x_sqa)
    # # plt.plot(fp_logit_acct_x_sqa,tp_logit_acct_x_sqa, 'b-', label ='acct_x_sqa features, AUC = %.3f'%area_logit_acct_x_sqa)
    # # plt.plot(fp_logit_accft_x_sqa,tp_logit_accft_x_sqa, 'g-', label ='accft_x_sqa features, AUC = %.3f'%area_logit_accft_x_sqa)
    # plt.plot(fp_logit_all,tp_logit_all, 'r-', label ='all features, AUC = %.3f'%area_logit_all)
    
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC of features using Logistic Regression (Training)')
    # plt.legend()
    # plt.show()
    
    
    
    
    ########
    ##b.test and validate
    #################################################################
    ##############################################
    # fp_test_logit_sqa, tp_test_logit_sqa, threshold_test_logit_sqa = roc_curve(pdv_1_vs_all_test, clf_logit_sqa.predict_proba(sqa_test_reshaped)[:,1])
    # area_test_logit_sqa = auc(fp_test_logit_sqa,tp_test_logit_sqa)
    
    # logit_sqa_result_test= pd.DataFrame(data=[["logit test sqa", sens_test_logit_sqa,spec_test_logit_sqa,prec_test_logit_sqa, area_test_logit_sqa]])
    
    # df_ML_results = pd.concat((df_ML_results, logit_sqa_result_test), axis=0,ignore_index=True)

    # fp_test_logit_accf, tp_test_logit_accf, threshold_test_logit_accf = roc_curve(pdv_1_vs_all_test, clf_logit_accf.predict_proba(accf_test_reshaped)[:,1])
    # area_test_logit_accf = auc(fp_test_logit_accf,tp_test_logit_accf)
    
    # logit_accf_result_test= pd.DataFrame(data=[["logit test accf", sens_test_logit_accf,spec_test_logit_accf,prec_test_logit_accf, area_test_logit_accf]]) 
    
    # df_ML_results = pd.concat((df_ML_results, logit_accf_result_test), ignore_index=True)
    

    # fp_test_logit_acct, tp_test_logit_acct, threshold_test_logit_acct = roc_curve(pdv_1_vs_all_test, clf_logit_acct.predict_proba(acct_test_reshaped)[:,1])
    # area_test_logit_acct = auc(fp_test_logit_acct,tp_test_logit_acct)
    
    # logit_acct_result_test= pd.DataFrame(data=[["logit test acct", sens_test_logit_acct,spec_test_logit_acct,prec_test_logit_acct, area_test_logit_acct]]) 
    
    # df_ML_results = pd.concat((df_ML_results, logit_acct_result_test), ignore_index=True)

    # fp_test_logit_accf_x_sqa, tp_test_logit_accf_x_sqa, threshold_test_logit_accf_x_sqa = roc_curve(pdv_1_vs_all_test, clf_logit_accf_x_sqa.predict_proba(df_test_2)[:,1])
    # area_test_logit_accf_x_sqa = auc(fp_test_logit_accf_x_sqa,tp_test_logit_accf_x_sqa)
    
    # logit_accf_x_sqa_result_test= pd.DataFrame(data=[["logit test accf_x_sqa", sens_test_logit_accf_x_sqa,spec_test_logit_accf_x_sqa,prec_test_logit_accf_x_sqa, area_test_logit_accf_x_sqa] ])
    
    # df_ML_results = pd.concat((df_ML_results, logit_accf_x_sqa_result_test), ignore_index=True)

    # fp_test_logit_acct_x_sqa, tp_test_logit_acct_x_sqa, threshold_test_logit_acct_x_sqa = roc_curve(pdv_1_vs_all_test, clf_logit_acct_x_sqa.predict_proba(df_test_3)[:,1])
    # area_test_logit_acct_x_sqa = auc(fp_test_logit_acct_x_sqa,tp_test_logit_acct_x_sqa)
    
    # logit_acct_x_sqa_result_test= pd.DataFrame(data=[["logit test acct_x_sqa", sens_test_logit_acct_x_sqa,spec_test_logit_acct_x_sqa,prec_test_logit_acct_x_sqa, area_test_logit_acct_x_sqa]]) 
    
    # df_ML_results = pd.concat((df_ML_results, logit_acct_x_sqa_result_test), ignore_index=True)

    # fp_test_logit_accft_x_sqa, tp_test_logit_accft_x_sqa, threshold_test_logit_accft_x_sqa = roc_curve(pdv_1_vs_all_test, clf_logit_accft_x_sqa.predict_proba(df_test_predictor_1)[:,1])
    # area_test_logit_accft_x_sqa = auc(fp_test_logit_accft_x_sqa,tp_test_logit_accft_x_sqa)
    
    # logit_accft_x_sqa_result_test= pd.DataFrame(data=[["logit test accft_x_sqa", sens_test_logit_accft_x_sqa,spec_test_logit_accft_x_sqa,prec_test_logit_accft_x_sqa, area_test_logit_accft_x_sqa]]) 
    
    # df_ML_results = pd.concat((df_ML_results, logit_accft_x_sqa_result_test), ignore_index=True)

    fp_test_logit_all, tp_test_logit_all, threshold_test_logit_all = roc_curve(pdv_1_vs_all_test, clf_logit_all.predict_proba(df_test_4)[:,1])
    area_test_logit_all = auc(fp_test_logit_all,tp_test_logit_all)
    
    logit_all_result_test= pd.DataFrame(data=[["logit test all", sens_test_logit_all,spec_test_logit_all,prec_test_logit_all, area_test_logit_all]])
    
    df_ML_results = pd.concat((df_ML_results, logit_all_result_test), ignore_index=True)

    # fp_test_sqa, tp_test_sqa, threshold_test_nn_sqa = roc_curve(pdv_1_vs_all_test, clf_nn_sqa.predict_proba(sqa_test_reshaped)[:,1])
    # area_test_nn_sqa = auc(fp_test_sqa,tp_test_sqa)
    
    # nn_sqa_result_test= pd.DataFrame(data=[["nn test sqa", sens_test_nn_sqa,spec_test_nn_sqa,prec_test_nn_sqa, area_test_nn_sqa]])
    
    # df_ML_results = pd.concat((df_ML_results, nn_sqa_result_test), ignore_index=True)

    # fp_test_accf, tp_test_accf, threshold_test_nn_accf = roc_curve(pdv_1_vs_all_test, clf_nn_accf.predict_proba(accf_test_reshaped)[:,1])
    # area_test_nn_accf = auc(fp_test_accf,tp_test_accf)
    
    # nn_accf_result_test= pd.DataFrame(data=[["nn test accf", sens_test_nn_accf,spec_test_nn_accf,prec_test_nn_accf, area_test_nn_accf]])
    
    # df_ML_results = pd.concat((df_ML_results, nn_accf_result_test), ignore_index=True)

    # fp_test_acct, tp_test_acct, threshold_test_nn_acct = roc_curve(pdv_1_vs_all_test, clf_nn_acct.predict_proba(acct_test_reshaped)[:,1])
    # area_test_nn_acct = auc(fp_test_acct,tp_test_acct)
    
    # nn_acct_result_test= pd.DataFrame(data=[["nn test acct", sens_test_nn_acct,spec_test_nn_acct,prec_test_nn_acct, area_test_nn_acct]])
    
    # df_ML_results = pd.concat((df_ML_results, nn_acct_result_test), ignore_index=True)

    # fp_test_accf_x_sqa, tp_test_accf_x_sqa, threshold_test_nn_accf_x_sqa = roc_curve(pdv_1_vs_all_test, clf_nn_accf_x_sqa.predict_proba(df_test_2)[:,1])
    # area_test_nn_accf_x_sqa = auc(fp_test_accf_x_sqa,tp_test_accf_x_sqa)
    
    # nn_accf_x_sqa_result_test= pd.DataFrame(data=[["nn test accf_x_sqa", sens_test_nn_accf_x_sqa,spec_test_nn_accf_x_sqa,prec_test_nn_accf_x_sqa, area_test_nn_accf_x_sqa]]) 
    
    # df_ML_results = pd.concat((df_ML_results, nn_accf_x_sqa_result_test), ignore_index=True)
    

    # fp_test_acct_x_sqa, tp_test_acct_x_sqa, threshold_test_nn_acct_x_sqa = roc_curve(pdv_1_vs_all_test, clf_nn_acct_x_sqa.predict_proba(df_test_3)[:,1])
    # area_test_nn_acct_x_sqa = auc(fp_test_acct_x_sqa,tp_test_acct_x_sqa)
    
    # nn_acct_x_sqa_result_test= pd.DataFrame(data=[["nn test acct_x_sqa", sens_test_nn_acct_x_sqa,spec_test_nn_acct_x_sqa,prec_test_nn_acct_x_sqa, area_test_nn_acct_x_sqa]]) 
    
    # df_ML_results = pd.concat((df_ML_results, nn_acct_x_sqa_result_test), ignore_index=True)
    

    # fp_test_accft_x_sqa, tp_test_accft_x_sqa, threshold_test_nn_accft_x_sqa = roc_curve(pdv_1_vs_all_test, clf_nn_accft_x_sqa.predict_proba(df_test_predictor_1)[:,1])
    # area_test_nn_accft_x_sqa = auc(fp_test_accft_x_sqa,tp_test_accft_x_sqa)
    
    # nn_accft_x_sqa_result_test= pd.DataFrame(data=[["nn test accft_x_sqa", sens_test_nn_accft_x_sqa,spec_test_nn_accft_x_sqa,prec_test_nn_accft_x_sqa, area_test_nn_accft_x_sqa]]) 
    
    # df_ML_results = pd.concat((df_ML_results, nn_accft_x_sqa_result_test), ignore_index=True)
    
    
    ###############################################################################
    ##################################
    
    fp_test_all, tp_test_all, threshold_test_nn_all = roc_curve(pdv_1_vs_all_test, clf_nn_all.predict_proba(df_test_4)[:,1])
    area_test_nn_all = auc(fp_test_all,tp_test_all)
    
    nn_all_result_test= pd.DataFrame(data=[["nn test all", sens_test_nn_all,spec_test_nn_all,prec_test_nn_all, area_test_nn_all]])
    
    df_ML_results = pd.concat((df_ML_results, nn_all_result_test), ignore_index=True)
    
    ######################################################################
    ######################################################################




    
    # plt.figure(3)
    # # plt.plot(fp_test_sqa,tp_test_sqa, 'm-', label ='sqa features, AUC = %.3f'%area_test_nn_sqa)
    # # plt.plot(fp_test_accf,tp_test_accf, 'c-', label ='accf features, AUC = %.3f'%area_test_nn_accf)
    # # plt.plot(fp_test_acct,tp_test_acct, 'k-', label ='acct features, AUC = %.3f'%area_test_nn_acct)
    # # plt.plot(fp_test_accf_x_sqa,tp_test_accf_x_sqa, 'y-', label ='accf_x_sqa features, AUC = %.3f'%area_test_nn_accf_x_sqa)
    # # plt.plot(fp_test_acct_x_sqa,tp_test_acct_x_sqa, 'b-', label ='acct_x_sqa features, AUC = %.3f'%area_test_nn_acct_x_sqa)
    # # plt.plot(fp_test_accft_x_sqa,tp_test_accft_x_sqa, 'g-', label ='accft_x_sqa features, AUC = %.3f'%area_test_nn_accft_x_sqa)
    # plt.plot(fp_test_all,tp_test_all, 'r-', label ='all features, AUC = %.3f'%area_test_nn_all)
    # #print(auc(fp_test_all, tp_test_all))
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC of features using NN (Test)')
    # plt.legend()
    # plt.show()
    
    # plt.figure(4)
    # # plt.plot(fp_test_logit_sqa,tp_test_logit_sqa, 'm-', label ='sqa features, AUC = %.3f'%area_test_logit_sqa)
    # # plt.plot(fp_test_logit_accf,tp_test_logit_accf, 'c-', label ='accf features, AUC = %.3f'%area_test_logit_accf)
    # # plt.plot(fp_test_logit_acct,tp_test_logit_acct, 'k-', label ='acct features, AUC = %.3f'%area_test_logit_acct)
    # # plt.plot(fp_test_logit_accf_x_sqa,tp_test_logit_accf_x_sqa, 'y-', label ='accf_x_sqa features, AUC = %.3f'%area_test_logit_accf_x_sqa)
    # # plt.plot(fp_test_logit_acct_x_sqa,tp_test_logit_acct_x_sqa, 'b-', label ='acct_x_sqa features, AUC = %.3f'%area_test_logit_acct_x_sqa)
    # # plt.plot(fp_test_logit_accft_x_sqa,tp_test_logit_accft_x_sqa, 'g-', label ='accft_x_sqa features, AUC = %.3f'%area_test_logit_accft_x_sqa)
    # plt.plot(fp_test_logit_all,tp_test_logit_all, 'r-', label ='all features, AUC = %.3f'%area_test_logit_all)
    
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC of features using Logistic Regression (Test)')
    # plt.legend()
    # plt.show()





    
# =============================================================================
#     all_coef = clf_nn_all.coefs_
#     all_features = clf_nn_all.feature_names_in_
# =============================================================================
    

    #print(clf_nn_all.coefs_)

    
    
    
    
    
    ##c. SVM SVC
    ###############################################################

    
    #PolynomialFeatures(2,)
    
    #accf_x_sqa = np.cross(accf,sqa_index)
    

    
    
    
    
    
    
    #i. sqa ##censored svc plot for speed modified here

    # clf_svc_sqa = SVC(C=1.0, kernel='poly', tol=1e-3, max_iter=2000, probability=True).fit(sqa_reshaped_scaled, pdv_1_vs_all)
    
    # cm_svc_sqa = confusion_matrix(pdv_1_vs_all, clf_svc_sqa.predict(sqa_reshaped_scaled))
    # sens_svc_sqa = cm_svc_sqa[1,1] / (cm_svc_sqa[1,1] + cm_svc_sqa[1,0])
    # spec_svc_sqa = cm_svc_sqa[0,0] / (cm_svc_sqa[0,0] + cm_svc_sqa[0,1])
    # prec_svc_sqa = cm_svc_sqa[1,1] / (cm_svc_sqa[1,1] + cm_svc_sqa[0,1])
    
    # cm_test_svc_sqa = confusion_matrix(pdv_1_vs_all_test, clf_svc_sqa.predict(sqa_test_reshaped_scaled))
    # sens_test_svc_sqa = cm_test_svc_sqa[1,1] / (cm_test_svc_sqa[1,1] + cm_test_svc_sqa[1,0])
    # spec_test_svc_sqa = cm_test_svc_sqa[0,0] / (cm_test_svc_sqa[0,0] + cm_test_svc_sqa[0,1])
    # prec_test_svc_sqa = cm_test_svc_sqa[1,1] / (cm_test_svc_sqa[1,1] + cm_test_svc_sqa[0,1])
    

    # fp_svc_sqa, tp_svc_sqa, threshold_svc_sqa = roc_curve(pdv_1_vs_all, clf_svc_sqa.predict_proba(sqa_reshaped_scaled)[:,1])
    # area_svc_sqa = auc(fp_svc_sqa,tp_svc_sqa)
    
    # svc_sqa_result= pd.DataFrame(data=[["svc sqa", sens_svc_sqa,spec_svc_sqa,prec_svc_sqa, area_svc_sqa]])
    
    # fp_test_svc_sqa, tp_test_svc_sqa, threshold_test_svc_sqa = roc_curve(pdv_1_vs_all_test, clf_svc_sqa.predict_proba(sqa_test_reshaped_scaled)[:,1])
    # area_test_svc_sqa = auc(fp_test_svc_sqa,tp_test_svc_sqa)
    
    # svc_sqa_result_test= pd.DataFrame(data=[["svc test sqa", sens_test_svc_sqa,spec_test_svc_sqa,prec_test_svc_sqa, area_test_svc_sqa]])
    
    
    
    # ##ii. accf
    # clf_svc_accf = SVC(C=1.0, kernel='poly', tol=1e-3, max_iter=2000, probability=True).fit(accf_reshaped_scaled, pdv_1_vs_all)
    
    # cm_svc_accf = confusion_matrix(pdv_1_vs_all, clf_svc_accf.predict(accf_reshaped_scaled))
    # sens_svc_accf = cm_svc_accf[1,1] / (cm_svc_accf[1,1] + cm_svc_accf[1,0])
    # spec_svc_accf = cm_svc_accf[0,0] / (cm_svc_accf[0,0] + cm_svc_accf[0,1])
    # prec_svc_accf = cm_svc_accf[1,1] / (cm_svc_accf[1,1] + cm_svc_accf[0,1])
    
    # cm_test_svc_accf = confusion_matrix(pdv_1_vs_all_test, clf_svc_accf.predict(accf_test_reshaped_scaled))
    # sens_test_svc_accf = cm_test_svc_accf[1,1] / (cm_test_svc_accf[1,1] + cm_test_svc_accf[1,0])
    # spec_test_svc_accf = cm_test_svc_accf[0,0] / (cm_test_svc_accf[0,0] + cm_test_svc_accf[0,1])
    # prec_test_svc_accf = cm_test_svc_accf[1,1] / (cm_test_svc_accf[1,1] + cm_test_svc_accf[0,1])
    

    # fp_svc_accf, tp_svc_accf, threshold_svc_accf = roc_curve(pdv_1_vs_all, clf_svc_accf.predict_proba(accf_reshaped_scaled)[:,1])
    # area_svc_accf = auc(fp_svc_accf,tp_svc_accf)
    
    # svc_accf_result= pd.DataFrame(data=[["svc accf", sens_svc_accf,spec_svc_accf,prec_svc_accf, area_svc_accf]])
    
    # fp_test_svc_accf, tp_test_svc_accf, threshold_test_svc_accf = roc_curve(pdv_1_vs_all_test, clf_svc_accf.predict_proba(accf_test_reshaped_scaled)[:,1])
    # area_test_svc_accf = auc(fp_test_svc_accf,tp_test_svc_accf)
    
    # svc_accf_result_test= pd.DataFrame(data=[["svc test accf", sens_test_svc_accf,spec_test_svc_accf,prec_test_svc_accf, area_test_svc_accf]])
    
    
    
    
    # clf_svc_acct = SVC(C=1.0, kernel='poly', tol=1e-3, max_iter=2000, probability=True).fit(acct_reshaped_scaled, pdv_1_vs_all)
    
    # cm_svc_acct = confusion_matrix(pdv_1_vs_all, clf_svc_acct.predict(acct_reshaped_scaled))
    # sens_svc_acct = cm_svc_acct[1,1] / (cm_svc_acct[1,1] + cm_svc_acct[1,0])
    # spec_svc_acct = cm_svc_acct[0,0] / (cm_svc_acct[0,0] + cm_svc_acct[0,1])
    # prec_svc_acct = cm_svc_acct[1,1] / (cm_svc_acct[1,1] + cm_svc_acct[0,1])
    
    # cm_test_svc_acct = confusion_matrix(pdv_1_vs_all_test, clf_svc_acct.predict(acct_test_reshaped_scaled))
    # sens_test_svc_acct = cm_test_svc_acct[1,1] / (cm_test_svc_acct[1,1] + cm_test_svc_acct[1,0])
    # spec_test_svc_acct = cm_test_svc_acct[0,0] / (cm_test_svc_acct[0,0] + cm_test_svc_acct[0,1])
    # prec_test_svc_acct = cm_test_svc_acct[1,1] / (cm_test_svc_acct[1,1] + cm_test_svc_acct[0,1])
    

    # fp_svc_acct, tp_svc_acct, threshold_svc_acct = roc_curve(pdv_1_vs_all, clf_svc_acct.predict_proba(acct_reshaped_scaled)[:,1])
    # area_svc_acct = auc(fp_svc_acct,tp_svc_acct)
    
    # svc_acct_result= pd.DataFrame(data=[["svc acct", sens_svc_acct,spec_svc_acct,prec_svc_acct, area_svc_acct]])
    
    # fp_test_svc_acct, tp_test_svc_acct, threshold_test_svc_acct = roc_curve(pdv_1_vs_all_test, clf_svc_acct.predict_proba(acct_test_reshaped_scaled)[:,1])
    # area_test_svc_acct = auc(fp_test_svc_acct,tp_test_svc_acct)
    
    # svc_acct_result_test= pd.DataFrame(data=[["svc test acct", sens_test_svc_acct,spec_test_svc_acct,prec_test_svc_acct, area_test_svc_acct]])
    

    
    
    # clf_svc_accf_x_sqa = SVC(C=1.0, kernel='poly', tol=1e-3, max_iter=2000, probability=True).fit(accf_x_sqa_reshaped_scaled, pdv_1_vs_all)
    
    # cm_svc_accf_x_sqa = confusion_matrix(pdv_1_vs_all, clf_svc_accf_x_sqa.predict(accf_x_sqa_reshaped_scaled))
    # sens_svc_accf_x_sqa = cm_svc_accf_x_sqa[1,1] / (cm_svc_accf_x_sqa[1,1] + cm_svc_accf_x_sqa[1,0])
    # spec_svc_accf_x_sqa = cm_svc_accf_x_sqa[0,0] / (cm_svc_accf_x_sqa[0,0] + cm_svc_accf_x_sqa[0,1])
    # prec_svc_accf_x_sqa = cm_svc_accf_x_sqa[1,1] / (cm_svc_accf_x_sqa[1,1] + cm_svc_accf_x_sqa[0,1])
    
    # cm_test_svc_accf_x_sqa = confusion_matrix(pdv_1_vs_all_test, clf_svc_accf_x_sqa.predict(accf_x_sqa_test_reshaped_scaled))
    # sens_test_svc_accf_x_sqa = cm_test_svc_accf_x_sqa[1,1] / (cm_test_svc_accf_x_sqa[1,1] + cm_test_svc_accf_x_sqa[1,0])
    # spec_test_svc_accf_x_sqa = cm_test_svc_accf_x_sqa[0,0] / (cm_test_svc_accf_x_sqa[0,0] + cm_test_svc_accf_x_sqa[0,1])
    # prec_test_svc_accf_x_sqa = cm_test_svc_accf_x_sqa[1,1] / (cm_test_svc_accf_x_sqa[1,1] + cm_test_svc_accf_x_sqa[0,1])
    

    # fp_svc_accf_x_sqa, tp_svc_accf_x_sqa, threshold_svc_accf_x_sqa = roc_curve(pdv_1_vs_all, clf_svc_accf_x_sqa.predict_proba(accf_x_sqa_reshaped_scaled)[:,1])
    # area_svc_accf_x_sqa = auc(fp_svc_accf_x_sqa,tp_svc_accf_x_sqa)
    
    # svc_accf_x_sqa_result= pd.DataFrame(data=[["svc accf_x_sqa", sens_svc_accf_x_sqa,spec_svc_accf_x_sqa,prec_svc_accf_x_sqa, area_svc_accf_x_sqa]])
    
    # fp_test_svc_accf_x_sqa, tp_test_svc_accf_x_sqa, threshold_test_svc_accf_x_sqa = roc_curve(pdv_1_vs_all_test, clf_svc_accf_x_sqa.predict_proba(accf_x_sqa_test_reshaped_scaled)[:,1])
    # area_test_svc_accf_x_sqa = auc(fp_test_svc_accf_x_sqa,tp_test_svc_accf_x_sqa)
    
    # svc_accf_x_sqa_result_test= pd.DataFrame(data=[["svc test accf_x_sqa", sens_test_svc_accf_x_sqa,spec_test_svc_accf_x_sqa,prec_test_svc_accf_x_sqa, area_test_svc_accf_x_sqa]])
    
    
    
    # clf_svc_acct_x_sqa = SVC(C=1.0, kernel='poly', tol=1e-3, max_iter=2000, probability=True).fit(acct_x_sqa_reshaped_scaled, pdv_1_vs_all)
    
    # cm_svc_acct_x_sqa = confusion_matrix(pdv_1_vs_all, clf_svc_acct_x_sqa.predict(acct_x_sqa_reshaped_scaled))
    # sens_svc_acct_x_sqa = cm_svc_acct_x_sqa[1,1] / (cm_svc_acct_x_sqa[1,1] + cm_svc_acct_x_sqa[1,0])
    # spec_svc_acct_x_sqa = cm_svc_acct_x_sqa[0,0] / (cm_svc_acct_x_sqa[0,0] + cm_svc_acct_x_sqa[0,1])
    # prec_svc_acct_x_sqa = cm_svc_acct_x_sqa[1,1] / (cm_svc_acct_x_sqa[1,1] + cm_svc_acct_x_sqa[0,1])
    
    # cm_test_svc_acct_x_sqa = confusion_matrix(pdv_1_vs_all_test, clf_svc_acct_x_sqa.predict(acct_x_sqa_test_reshaped_scaled))
    # sens_test_svc_acct_x_sqa = cm_test_svc_acct_x_sqa[1,1] / (cm_test_svc_acct_x_sqa[1,1] + cm_test_svc_acct_x_sqa[1,0])
    # spec_test_svc_acct_x_sqa = cm_test_svc_acct_x_sqa[0,0] / (cm_test_svc_acct_x_sqa[0,0] + cm_test_svc_acct_x_sqa[0,1])
    # prec_test_svc_acct_x_sqa = cm_test_svc_acct_x_sqa[1,1] / (cm_test_svc_acct_x_sqa[1,1] + cm_test_svc_acct_x_sqa[0,1])
    

    # fp_svc_acct_x_sqa, tp_svc_acct_x_sqa, threshold_svc_acct_x_sqa = roc_curve(pdv_1_vs_all, clf_svc_acct_x_sqa.predict_proba(acct_x_sqa_reshaped_scaled)[:,1])
    # area_svc_acct_x_sqa = auc(fp_svc_acct_x_sqa,tp_svc_acct_x_sqa)
    
    # svc_acct_x_sqa_result= pd.DataFrame(data=[["svc acct_x_sqa", sens_svc_acct_x_sqa,spec_svc_acct_x_sqa,prec_svc_acct_x_sqa, area_svc_acct_x_sqa]])
    
    # fp_test_svc_acct_x_sqa, tp_test_svc_acct_x_sqa, threshold_test_svc_acct_x_sqa = roc_curve(pdv_1_vs_all_test, clf_svc_acct_x_sqa.predict_proba(acct_x_sqa_test_reshaped_scaled)[:,1])
    # area_test_svc_acct_x_sqa = auc(fp_test_svc_acct_x_sqa,tp_test_svc_acct_x_sqa)
    
    # svc_acct_x_sqa_result_test= pd.DataFrame(data=[["svc test acct_x_sqa", sens_test_svc_acct_x_sqa,spec_test_svc_acct_x_sqa,prec_test_svc_acct_x_sqa, area_test_svc_acct_x_sqa]])
    
    
    
    
    # clf_svc_accft_x_sqa = SVC(C=1.0, kernel='poly', tol=1e-3, max_iter=2000, probability=True).fit(accft_x_sqa_reshaped_scaled, pdv_1_vs_all)
    
    # cm_svc_accft_x_sqa = confusion_matrix(pdv_1_vs_all, clf_svc_accft_x_sqa.predict(accft_x_sqa_reshaped_scaled))
    # sens_svc_accft_x_sqa = cm_svc_accft_x_sqa[1,1] / (cm_svc_accft_x_sqa[1,1] + cm_svc_accft_x_sqa[1,0])
    # spec_svc_accft_x_sqa = cm_svc_accft_x_sqa[0,0] / (cm_svc_accft_x_sqa[0,0] + cm_svc_accft_x_sqa[0,1])
    # prec_svc_accft_x_sqa = cm_svc_accft_x_sqa[1,1] / (cm_svc_accft_x_sqa[1,1] + cm_svc_accft_x_sqa[0,1])
    
    # cm_test_svc_accft_x_sqa = confusion_matrix(pdv_1_vs_all_test, clf_svc_accft_x_sqa.predict(accft_x_sqa_test_reshaped_scaled))
    # sens_test_svc_accft_x_sqa = cm_test_svc_accft_x_sqa[1,1] / (cm_test_svc_accft_x_sqa[1,1] + cm_test_svc_accft_x_sqa[1,0])
    # spec_test_svc_accft_x_sqa = cm_test_svc_accft_x_sqa[0,0] / (cm_test_svc_accft_x_sqa[0,0] + cm_test_svc_accft_x_sqa[0,1])
    # prec_test_svc_accft_x_sqa = cm_test_svc_accft_x_sqa[1,1] / (cm_test_svc_accft_x_sqa[1,1] + cm_test_svc_accft_x_sqa[0,1])
    

    # fp_svc_accft_x_sqa, tp_svc_accft_x_sqa, threshold_svc_accft_x_sqa = roc_curve(pdv_1_vs_all, clf_svc_accft_x_sqa.predict_proba(accft_x_sqa_reshaped_scaled)[:,1])
    # area_svc_accft_x_sqa = auc(fp_svc_accft_x_sqa,tp_svc_accft_x_sqa)
    
    # svc_accft_x_sqa_result= pd.DataFrame(data=[["svc accft_x_sqa", sens_svc_accft_x_sqa,spec_svc_accft_x_sqa,prec_svc_accft_x_sqa, area_svc_accft_x_sqa]])
    
    # fp_test_svc_accft_x_sqa, tp_test_svc_accft_x_sqa, threshold_test_svc_accft_x_sqa = roc_curve(pdv_1_vs_all_test, clf_svc_accft_x_sqa.predict_proba(accft_x_sqa_test_reshaped_scaled)[:,1])
    # area_test_svc_accft_x_sqa = auc(fp_test_svc_accft_x_sqa,tp_test_svc_accft_x_sqa)
    
    # svc_accft_x_sqa_result_test= pd.DataFrame(data=[["svc test accft_x_sqa", sens_test_svc_accft_x_sqa,spec_test_svc_accft_x_sqa,prec_test_svc_accft_x_sqa, area_test_svc_accft_x_sqa]])
    
    
    
    
    clf_svc_all = SVC(C=1, kernel='poly', tol=1e-3, max_iter=3000, probability=True).fit(df_4, pdv_1_vs_all)
    
    pred_nd_svc = clf_svc_all.predict(df_4_nodrop)
    df_diff_svc = pd.DataFrame()
    loopcounter=0
    for i in pdvnd_1_vs_all.index:
        if ((pdvnd_1_vs_all[i]!=1) and (pred_nd_svc[loopcounter]==1)):
            df_diff_svc = pd.concat((df_diff_svc, pd.Series([i])))
        loopcounter+=1
            
    pred_nd_svc_test = clf_svc_all.predict(df_test_4_nodrop)     
    loopcounter=0  
    df_diff_svc = pd.DataFrame()
    for i in pdvnd_1_vs_all_test.index:
        if ((pdvnd_1_vs_all_test[i]!=1) and (pred_nd_svc_test[loopcounter]==1)):
            df_diff_svc = pd.concat((df_diff_svc, pd.Series([i])))
        loopcounter+=1
    
    cm_svc_all = confusion_matrix(pdv_1_vs_all, clf_svc_all.predict(df_4))
    sens_svc_all = cm_svc_all[1,1] / (cm_svc_all[1,1] + cm_svc_all[1,0])
    spec_svc_all = cm_svc_all[0,0] / (cm_svc_all[0,0] + cm_svc_all[0,1])
    prec_svc_all = cm_svc_all[1,1] / (cm_svc_all[1,1] + cm_svc_all[0,1])
    
    cm_test_svc_all = confusion_matrix(pdv_1_vs_all_test, clf_svc_all.predict(df_test_4))
    sens_test_svc_all = cm_test_svc_all[1,1] / (cm_test_svc_all[1,1] + cm_test_svc_all[1,0])
    spec_test_svc_all = cm_test_svc_all[0,0] / (cm_test_svc_all[0,0] + cm_test_svc_all[0,1])
    prec_test_svc_all = cm_test_svc_all[1,1] / (cm_test_svc_all[1,1] + cm_test_svc_all[0,1])
    

    fp_svc_all, tp_svc_all, threshold_svc_all = roc_curve(pdv_1_vs_all, clf_svc_all.predict_proba(df_4)[:,1])
    area_svc_all = auc(fp_svc_all,tp_svc_all)
    
    svc_all_result= pd.DataFrame(data=[["svc all", sens_svc_all,spec_svc_all,prec_svc_all, area_svc_all]])
    
    fp_test_svc_all, tp_test_svc_all, threshold_test_svc_all = roc_curve(pdv_1_vs_all_test, clf_svc_all.predict_proba(all_test_reshaped_scaled)[:,1])
    area_test_svc_all = auc(fp_test_svc_all,tp_test_svc_all)
    
    svc_all_result_test= pd.DataFrame(data=[["svc test all", sens_test_svc_all,spec_test_svc_all,prec_test_svc_all, area_test_svc_all]])
    
    plt.figure(10)
    perm_importance_svc = permutation_importance(clf_svc_all, df_test_4, pdv_1_vs_all_test)

    sorted_idx = perm_importance_svc.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance_svc.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance (SVC)")
    plt.show()
    
    #print(clf_svc_all.coef_)
    
    
    # df_ML_results = pd.concat((df_ML_results, svc_sqa_result), axis=0,ignore_index=True)
    
    # df_ML_results = pd.concat((df_ML_results, svc_sqa_result_test), axis=0,ignore_index=True)
    
    
    # df_ML_results = pd.concat((df_ML_results, svc_accf_result), axis=0,ignore_index=True)
    
    # df_ML_results = pd.concat((df_ML_results, svc_accf_result_test), axis=0,ignore_index=True)
    
    
    # df_ML_results = pd.concat((df_ML_results, svc_acct_result), axis=0,ignore_index=True)
    
    # df_ML_results = pd.concat((df_ML_results, svc_acct_result_test), axis=0,ignore_index=True)
    
    
    # df_ML_results = pd.concat((df_ML_results, svc_accf_x_sqa_result), axis=0,ignore_index=True)
    
    # df_ML_results = pd.concat((df_ML_results, svc_accf_x_sqa_result_test), axis=0,ignore_index=True)
    
    # df_ML_results = pd.concat((df_ML_results, svc_acct_x_sqa_result), axis=0,ignore_index=True)
    
    # df_ML_results = pd.concat((df_ML_results, svc_acct_x_sqa_result_test), axis=0,ignore_index=True)
    
    # df_ML_results = pd.concat((df_ML_results, svc_accft_x_sqa_result), axis=0,ignore_index=True)
    
    # df_ML_results = pd.concat((df_ML_results, svc_accft_x_sqa_result_test), axis=0,ignore_index=True)
    
    df_ML_results = pd.concat((df_ML_results, svc_all_result), axis=0,ignore_index=True)
    
    df_ML_results = pd.concat((df_ML_results, svc_all_result_test), axis=0,ignore_index=True)
    
    
    
    # plt.figure(5)
    # # plt.plot(fp_svc_sqa,tp_svc_sqa, 'm-', label ='sqa features, AUC = %.3f'%area_svc_sqa)
    # # plt.plot(fp_svc_accf,tp_svc_accf, 'c-', label ='accf features, AUC = %.3f'%area_svc_accf)
    # # plt.plot(fp_svc_acct,tp_svc_acct, 'k-', label ='acct features, AUC = %.3f'%area_svc_acct)
    # # plt.plot(fp_svc_accf_x_sqa,tp_svc_accf_x_sqa, 'y-', label ='accf_x_sqa features, AUC = %.3f'%area_svc_accf_x_sqa)
    # # plt.plot(fp_svc_acct_x_sqa,tp_svc_acct_x_sqa, 'b-', label ='acct_x_sqa features, AUC = %.3f'%area_svc_acct_x_sqa)
    # # plt.plot(fp_svc_accft_x_sqa,tp_svc_accft_x_sqa, 'g-', label ='accft_x_sqa features, AUC = %.3f'%area_svc_accft_x_sqa)
    # plt.plot(fp_svc_all,tp_svc_all, 'r-', label ='all features, AUC = %.3f'%area_svc_all)
    
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC of features using SVC (Training)')
    # plt.legend()
    # plt.show()
    

    # plt.figure(6)
    # # plt.plot(fp_test_svc_sqa,tp_test_svc_sqa, 'm-', label ='sqa features, AUC = %.3f'%area_test_svc_sqa)
    # # plt.plot(fp_test_svc_accf,tp_test_svc_accf, 'c-', label ='accf features, AUC = %.3f'%area_test_svc_accf)
    # # plt.plot(fp_test_svc_acct,tp_test_svc_acct, 'k-', label ='acct features, AUC = %.3f'%area_test_svc_acct)
    # # plt.plot(fp_test_svc_accf_x_sqa,tp_test_svc_accf_x_sqa, 'y-', label ='accf_x_sqa features, AUC = %.3f'%area_test_svc_accf_x_sqa)
    # # plt.plot(fp_test_svc_acct_x_sqa,tp_test_svc_acct_x_sqa, 'b-', label ='acct_x_sqa features, AUC = %.3f'%area_test_svc_acct_x_sqa)
    # # plt.plot(fp_test_svc_accft_x_sqa,tp_test_svc_accft_x_sqa, 'g-', label ='accft_x_sqa features, AUC = %.3f'%area_test_svc_accft_x_sqa)
    # plt.plot(fp_test_svc_all,tp_test_svc_all, 'r-', label ='all features, AUC = %.3f'%area_test_svc_all)
    
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC of features using SVC (Test)')
    # plt.legend()
    # plt.show()
    
    
    
    
    
    #print(clf_svc_all.class_weight_, clf_svc_all.classes_, clf_svc_all.n_features_in_)
    #aaaaaaa= clf_svc_all.dual_coef_
    #aaaab = clf_svc_all.support_vectors_
    #aac = clf_svc_all.decision_function(all_reshaped_scaled)
    

# =============================================================================
    plt.figure(7)
    plt.plot(fp_all,tp_all, 'r-', label ='NN, AUC = %.3f'%area_nn_all)
    plt.plot(fp_svc_all,tp_svc_all, 'g-', label ='SVC, AUC = %.3f'%area_svc_all)
    plt.plot(fp_logit_all,tp_logit_all, 'b-', label ='Logit, AUC = %.3f'%area_logit_all)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparison of all ML technique\'s ROC, Training')
    plt.legend()
    plt.show()
#     

    plt.figure(8)
    plt.plot(fp_test_all,tp_test_all, 'r-', label ='NN, AUC = %.3f'%area_test_nn_all)
    plt.plot(fp_test_svc_all,tp_test_svc_all, 'g-', label ='SVC, AUC = %.3f'%area_test_svc_all)
    plt.plot(fp_test_logit_all,tp_test_logit_all, 'b-', label ='Logit, AUC = %.3f'%area_test_logit_all)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparison of all ML technique\'s ROC, Test')
    plt.legend()
    plt.show()

# =============================================================================

df_ML_results = df_ML_results.rename(columns={0:'Features',1:'Sens', 2:'Spec', 3:'Prec', 4:'ROC AUC'})    
    
    
    
    
    #predicted vs true plot 
    #########################################################

# =============================================================================
#     aaaaaaa=clf_nn_all.predict(df_4)
#     
 #=============================================================================

