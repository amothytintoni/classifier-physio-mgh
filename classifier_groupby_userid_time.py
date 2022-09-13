import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import graphviz as gv
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
import functions_for_classifier as funcs
import time
#import correct_fnfp as corrector
import warnings
warnings.filterwarnings('ignore', '.*do not.*')
import datetime as dt


pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.3f}".format

patient_grouping=True
samplefrac=1.0
trycount= 1
utc_offset = [-4, 0]
buckets_count = 300
threshold_bins_count = 300
output_threshold = 0.50
correction_flag = -1
scale_imbalanced_class = 1.0/1.11
#-1 if all fn,fp correction #1 if fp correction only, 0 if no correction #2 if fn correction only.

#enter any labelling correction file
#corr = pd.read_csv('Book1.csv')
#corr = corr.drop(columns = ['remark', 'motion validity'])
######################################################################
####################################################################


final_layer_threshold = np.linspace(0.5, 1.0, 21)
counter_main_loop = 0



print("output_threshold = {}".format(output_threshold))
print("correction_flag =",correction_flag)
if correction_flag==0:
    path_dataset = os.path.join('C:\\Users\\Timothy Antoni\\Documents\\programming\\Dataset', 'final1')
elif correction_flag!=0:
    path_dataset = os.path.join('C:\\Users\\Timothy Antoni\\Documents\\','programming\\fnfpcheck')
#path_dataset = os.path.join('C:\\Users\\Timothy Antoni\\Documents\\programming', 'fpfnplot')

ignored = {'undone', 'sus', 'before_2', 'test','before_15_8'}
list_file = [x for x in os.listdir(path_dataset) if x not in ignored]
list_file = [x for x in list_file if x.endswith('.csv')]
#print(list_file)

############################
############### if got any correction (fn, fp)
if correction_flag ==-1:
    df = pd.read_csv(os.path.join(path_dataset,'df_corrected_fnfp.csv'))
    # df = pd.read_csv(os.path.join(path_dataset,'corr2_for_prog_final.csv'))
    df = df[['userId','dateTime', 
    'PD Validity',                                                                                                'Accl Validity',
        'point_awake', 'point_sleep',
        'SQA_index', 
        'accepted_frame_ratio', 
        # 'accepted_frame_tdomain_ratio', 
        'list_sensor_onskin_status_ratio',
        'val_sd_signal_w_sqa',
        'list_onskin_status_stdev_ratio',
            'skin_temperature',
            'timeLengthValidity', 
            'dashboardMode',
            ]]


if correction_flag ==2:
    df = pd.read_csv(os.path.join(path_dataset,'df_corrected_fn.csv'))
    df = df[['userId','dateTime', 
    'PD Validity',                                                                                                'Accl Validity',
        'point_awake', 'point_sleep',
        'SQA_index', 
        'accepted_frame_ratio', 
        # 'accepted_frame_tdomain_ratio', 
        'list_sensor_onskin_status_ratio',
        'val_sd_signal_w_sqa',
        'list_onskin_status_stdev_ratio',
            'skin_temperature',
            'timeLengthValidity', 
            'dashboardMode',
            ]]

if correction_flag==1:
    df = pd.read_csv(os.path.join(path_dataset,'df_corrected_fp.csv'))
    df = df[['userId','dateTime', 
    'PD Validity',                                                                                                'Accl Validity',
        'point_awake', 'point_sleep',
        'SQA_index', 
        'accepted_frame_ratio', 
        # 'accepted_frame_tdomain_ratio', 
        'list_sensor_onskin_status_ratio',
        'val_sd_signal_w_sqa',
        'list_onskin_status_stdev_ratio',
            'skin_temperature',
            'timeLengthValidity', 
            'dashboardMode',
            ]]
############

if correction_flag==0:
    df=pd.DataFrame()
    for filename in list_file:    
        read_data = pd.read_csv(os.path.join(path_dataset, filename), encoding = "ISO-8859-1")[['userId','dateTime', 
    'PD Validity',                                                                                                'Accl Validity',
        'point_awake', 'point_sleep',
        'SQA_index', 
        'accepted_frame_ratio', 
        # 'accepted_frame_tdomain_ratio', 
        'list_sensor_onskin_status_ratio',
        'val_sd_signal_w_sqa',
        'list_onskin_status_stdev_ratio',
            'skin_temperature',
            'timeLengthValidity',
            'dashboardMode',
        # 'RR_DC', 'RR_fdomain_w_good_sqa', 'RR_IBI', 'RR_TD', 'RR_tdomain', 'RR',
        #'sensor_onskin_status'
        ]]
        
        #print(filename, len(read_data))
        df = pd.concat((df, read_data),ignore_index = True)

feature_names = [ 
                  'SQA_index', 
                  'accepted_frame_ratio', 
                  # 'accepted_frame_tdomain_ratio', 
                  'list_sensor_onskin_status_ratio',
                   'val_sd_signal_w_sqa',
                  'list_onskin_status_stdev_ratio',
                  'skin_temperature',
                  'activity_level'
                  ]
features = np.array(feature_names)

for col in df.columns:
    df = df[df[col].notna()]

cols=[]
for col in df.columns:
    cols = np.append(cols, col)
# date_object = datetime.strptime(date_string, "%d %B, %Y")

# print("before RR mode filter", df.shape)
df = df[df['dashboardMode']=='RR']
df = df.drop(columns=['dashboardMode'])
# print("after RR mode filter", df.shape)

#######datetime only
#########

# for i in df.index:
#     df['dateTime'][i] = dt.datetime.strptime(df['dateTime'][i], "%Y-%m-%d %H:%M:%S")
#     #print("type of date_object =", type(df['dateTime'][i]))
# ndf = dict(tuple(df.groupby(pd.Grouper(key="dateTime", freq="1D"))))
# index_ndf = list(ndf)

# df_train_dtg = pd.DataFrame()
# df_test_dtg = pd.DataFrame()
# train_frac = 0.6


# for i in range(0,int(len(index_ndf)*train_frac)):
#     df_train_dtg = pd.concat((df_train_dtg, ndf[index_ndf[i]]))

# #df_train_dtg = df_train_dtg[1]

# for i in range(int(len(index_ndf)*train_frac), int(len(index_ndf))):
#     df_test_dtg = pd.concat((df_test_dtg, ndf[index_ndf[i]]))
    
    
    
    
    
    
    
# df_test_dtg = df_test_dtg[1]
# dfs = [ndf.get_group(x) for x in ndf.groups]


# ndf = pd.DataFrame()
# for col in cols:
#     ndf = pd.concat((df.{}.format(col), col),axis=1)


#patient based
patient_ids = []

for i in df.index:
    if df['userId'][i] not in patient_ids:
        patient_ids = np.append(patient_ids, df['userId'][i])
# print("no of patients =", len(patient_ids))


df_train_ppt = pd.DataFrame() 
df_test_ppt = pd.DataFrame() 
df_nodrop_train_ppt = pd.DataFrame() 
df_nodrop_test_ppt = pd.DataFrame() 
for i in patient_ids:
    df_p1 = df[df['userId']==i]
    #df_p1 = df.sort_values('dateTime')
    
    #########train on earlier 60%
    # df_p1_train, df_p1_test = np.split(df_p1, [int(.6*len(df_p1))])
    # if i==152:
    #     print("training on first 60%")
    
    ######train on last 60%
    # df_p1_test, df_p1_train = np.split(df_p1, [int(.4*len(df_p1))])
    # if i==152:
    #     print("training on last 60%")
    
    #####train -test split is randomized random 60 (only patient based, not time)
    df_p1_train, df_p1_test = train_test_split(df_p1, test_size=0.4, shuffle=True)
    if i==152:
        print("training on random 60%")
    
    
    df_train_ppt = pd.concat((df_train_ppt,df_p1_train),ignore_index=True)
    df_test_ppt = pd.concat((df_test_ppt,df_p1_test),ignore_index=True)
    df_nodrop_train_ppt = pd.concat((df_nodrop_train_ppt,df_p1_train))
    df_nodrop_test_ppt = pd.concat((df_nodrop_test_ppt,df_p1_test))
 
patient_grouping =  True
print("patient_grouping =",patient_grouping)
if patient_grouping ==True:
    df = df_train_ppt.copy()
    df_test = df_test_ppt.copy()
    df_nodrop = df_nodrop_train_ppt.copy()
    df_nodrop_test = df_nodrop_test_ppt.copy()  
    # df = df_p1_train
    # df_test = df_p1_test
    # df_nodrop = df
    # df_nodrop_test = df_test

df = df.drop(columns=['userId','dateTime'])
df_nodrop = df_nodrop.drop(columns=['userId','dateTime'])
df_test = df_test.drop(columns=['userId','dateTime'])
df_nodrop_test = df_nodrop_test.drop(columns=['userId','dateTime'])


df = df[df['timeLengthValidity']==1].reset_index(drop=True)
df = df.drop(columns='timeLengthValidity')
df_nodrop = df_nodrop[df_nodrop['timeLengthValidity']==1]
df_nodrop = df_nodrop.drop(columns='timeLengthValidity')
df_test = df_test[df_test['timeLengthValidity']==1].reset_index(drop=True)
df_test = df_test.drop(columns='timeLengthValidity')
df_nodrop_test = df_nodrop_test[df_nodrop_test['timeLengthValidity']==1]
df_nodrop_test = df_nodrop_test.drop(columns='timeLengthValidity')

df['activity_level'] = ((df['point_awake'])/(df['point_sleep']+df['point_awake']))
df_nodrop['activity_level'] = ((df_nodrop['point_awake'])/(df_nodrop['point_sleep']+df_nodrop['point_awake']))
df_test['activity_level'] = ((df_test['point_awake'])/(df_test['point_sleep']+df_test['point_awake']))
df_nodrop_test['activity_level'] = ((df_nodrop_test['point_awake'])/(df_nodrop_test['point_sleep']+df_nodrop_test['point_awake']))

df = df.reset_index(drop=True)

pdv = df['PD Validity'].reset_index(drop=True)
pdv_test = df_test['PD Validity'].reset_index(drop=True)
pdvnd = df_nodrop['PD Validity']
pdvnd_test = df_nodrop_test['PD Validity']

pdv_1_vs_all = pdv.copy()
pdv_0_vs_all = pdv.copy()
pdvnd_1_vs_all = pdvnd.copy()
pdvnd_0_vs_all = pdvnd.copy()
pdv_1_vs_all_test = pdv_test.copy()
pdv_0_vs_all_test = pdv_test.copy()
pdvnd_1_vs_all_test = pdvnd_test.copy()
pdvnd_0_vs_all_test = pdvnd_test.copy()

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

# print("train pd0 =",len(df[df['PD Validity']==0]))
# print("train pd1 =",len(df[df['PD Validity']==1]))
# print("train pd2 =",len(df[df['PD Validity']==2]))
# print("test pd0 =",len(df_test[df_test['PD Validity']==0]))
# print("test pd1 =",len(df_test[df_test['PD Validity']==1]))
# print("test pd2 =",len(df_test[df_test['PD Validity']==2]))

df_4 = df.drop(columns=df.iloc[:,0:4])
df_4_nodrop = df_nodrop.drop(columns=df.iloc[:,0:4])

df_test_4 = df_test.drop(columns=df_test.iloc[:,0:4])
df_test_4_nodrop = df_nodrop_test.drop(columns=df.iloc[:,0:4])

df_ML_results = pd.DataFrame()
df_fn_results = pd.DataFrame()
df_fp_results = pd.DataFrame()

print("train data length =", df.shape[0])
print("test data length =", df_test.shape[0])


# =============================================================================
    #preprocessing prepro

pdv_1_vs_all_numpy = pdv_1_vs_all.to_numpy(copy=True)
pdv_1_vs_all_reshaped = np.reshape(pdv_1_vs_all_numpy,(-1,1))

pdv_1_vs_all_test_numpy = pdv_1_vs_all_test.to_numpy(copy=True)
pdv_1_vs_all_test_reshaped = np.reshape(pdv_1_vs_all_test_numpy,(-1,1))

all_reshaped_scaled = MinMaxScaler().fit_transform(df_4)

all_test_reshaped_scaled = MinMaxScaler().fit_transform(df_test_4)


all_reshaped = all_reshaped_scaled

all_test_reshaped = all_test_reshaped_scaled

df_4 = MinMaxScaler().fit_transform(df_4)
df_test_4 = MinMaxScaler().fit_transform(df_test_4)
df_4_nodrop = MinMaxScaler().fit_transform(df_4_nodrop)
df_test_4_nodrop = MinMaxScaler().fit_transform(df_test_4_nodrop)

for i in range(0,trycount):
    
    #################xgboost xgb
    ###################
    dtrain = xgb.DMatrix(df_4,label=pdv_1_vs_all)
    dtrain_raw = xgb.DMatrix(df_4)
    dtest = xgb.DMatrix(df_test_4, label= pdv_1_vs_all_test)
    num_round=4000
    xgb_param = {'max_depth':8, 
                 'alpha':50,
                 'eta':0.01,
                 'gamma':0,
                 'scale_pos_weight': 1.11,
                 # 'lambda': 4,
                 'objective': 'binary:logistic'
                     
                 }
    xgb_param['nthread'] = 4
    xgb_param['eval_metric'] = ['auc']
    
    print(xgb_param)
    
    evallist = [(dtest,'eval'), (dtrain, 'train')]
    
    bst = xgb.train(xgb_param, 
                    dtrain, 
                    num_round, 
                    #evallist, 
                    #save_name="xgbmodel"
                    )
    # bst.save_model('xgb1.model')
    
    xgb.plot_importance(bst)
    
    pred_train_xgb_raw = bst.predict(dtrain)
    pred_test_xgb_raw = bst.predict(dtest)
    pred_train_xgb = pred_train_xgb_raw.copy()
    pred_test_xgb = pred_test_xgb_raw.copy()
    
    pred_nd_xgb = pred_train_xgb
    df_diff_xgb = pd.DataFrame()
    loopcounter=0
    for i in pdvnd_1_vs_all.index:
        # if ((pdvnd_1_vs_all[i]!=1) and (pred_nd_xgb[loopcounter]==1)):
        if (pdvnd_1_vs_all[i] != pred_nd_xgb[loopcounter]):
            df_diff_xgb = pd.concat((df_diff_xgb, pd.Series([i])))
        loopcounter+=1
            
    pred_nd_xgb_test = pred_test_xgb     
    loopcounter=0
    df_diff_xgb = pd.DataFrame()
    for i in pdvnd_1_vs_all_test.index:
        # if ((pdvnd_1_vs_all_test[i]!=1) and (pred_nd_xgb_test[loopcounter]==1)):
        if (pdvnd_1_vs_all_test[i] != pred_nd_xgb_test[loopcounter]):
            df_diff_xgb = pd.concat((df_diff_xgb, pd.Series([i])))
        loopcounter+=1


    for i in range(0,len(pred_train_xgb)):
        if pred_train_xgb[i] >output_threshold:
            pred_train_xgb[i] = 1
        else:
            pred_train_xgb[i] = 0
            
    for i in range(0,len(pred_test_xgb)):
        if pred_test_xgb[i] >output_threshold:
            pred_test_xgb[i] = 1
        else:
            pred_test_xgb[i] = 0
    
    
    cm_xgb = confusion_matrix(pdv_1_vs_all, pred_train_xgb)
    sens_xgb = cm_xgb[1,1] / (cm_xgb[1,1] + cm_xgb[1,0])
    spec_xgb = cm_xgb[0,0] / (cm_xgb[0,0] + cm_xgb[0,1])
    prec_xgb = cm_xgb[1,1] / (cm_xgb[1,1] + cm_xgb[0,1])
    
    cm_test_xgb = confusion_matrix(pdv_1_vs_all_test, pred_test_xgb)
    sens_test_xgb = cm_test_xgb[1,1] / (cm_test_xgb[1,1] + cm_test_xgb[1,0])
    spec_test_xgb = cm_test_xgb[0,0] / (cm_test_xgb[0,0] + cm_test_xgb[0,1])
    prec_test_xgb = cm_test_xgb[1,1] / (cm_test_xgb[1,1] + cm_test_xgb[0,1])
    
    # fp_xgb, tp_xgb, threshold_xgb = roc_curve(pdv_1_vs_all, pred_train_xgb)
    # area_xgb = auc(fp_xgb,tp_xgb)
    fp_xgb, tp_xgb, threshold_xgb = funcs.my_roc(pdv_1_vs_all, pred_train_xgb_raw, threshold_bins_count)
    area_xgb = auc(fp_xgb,tp_xgb)
    
    xgb_result= pd.DataFrame(data=[["xgb", sens_xgb,spec_xgb,prec_xgb, area_xgb]])
    
    df_ML_results = pd.concat((df_ML_results, xgb_result), ignore_index=True)
    
    fp_test_xgb, tp_test_xgb, threshold_test_xgb = funcs.my_roc(pdv_1_vs_all_test, pred_test_xgb_raw, threshold_bins_count)
    area_test_xgb = auc(fp_test_xgb,tp_test_xgb)
    
    xgb_result_test= pd.DataFrame(data=[["xgb test", sens_test_xgb,spec_test_xgb,prec_test_xgb, area_test_xgb]])
    
    df_ML_results = pd.concat((df_ML_results, xgb_result_test), ignore_index=True)
    
    
    pred_nd_xgb = pred_train_xgb.copy()
    df_diff_xgb = pd.DataFrame()
    loopcounter=0
    for i in pdvnd_1_vs_all.index:
        if ((pdvnd_1_vs_all[i]==1) and (pred_nd_xgb[loopcounter]!=1)):
            df_diff_xgb = pd.concat((df_diff_xgb, pd.Series([i])))
        loopcounter+=1
            
    pred_nd_xgb_test = pred_test_xgb.copy() 
    
    loopcounter=0  
    
    for i in pdvnd_1_vs_all_test.index:
        if ((pdvnd_1_vs_all_test[i]==1) and (pred_nd_xgb_test[loopcounter]!=1)):
            df_diff_xgb = pd.concat((df_diff_xgb, pd.Series([i])))
        loopcounter+=1
    
    df_fn_results = df_diff_xgb.copy()
    
    pred_nd_xgb = pred_train_xgb.copy()
    
    df_diff_xgb = pd.DataFrame()
    loopcounter=0
    for i in pdvnd_1_vs_all.index:
        if ((pdvnd_1_vs_all[i]!=1) and (pred_nd_xgb[loopcounter]==1)):
            df_diff_xgb = pd.concat((df_diff_xgb, pd.Series([i])))
        loopcounter+=1
            
    pred_nd_xgb_test = pred_test_xgb.copy()    
    loopcounter=0  

    for i in pdvnd_1_vs_all_test.index:
        if ((pdvnd_1_vs_all_test[i]!=1) and (pred_nd_xgb_test[loopcounter]==1)):
            df_diff_xgb = pd.concat((df_diff_xgb, pd.Series([i])))
        loopcounter+=1
    
    df_fp_results = df_diff_xgb.copy()
    
    df_diff_xgb = pd.concat((df_fp_results, df_fn_results))
    
    #print(clf_xgb.coef_)
    # plt.figure(12)
    #perm_importance_xgb = permutation_importance(clf_xgb, df_test_4, pdv_1_vs_all_test)

    # sorted_idx = perm_importance_xgb.importances_mean.argsort()
    # plt.barh(features[sorted_idx], perm_importance_xgb.importances_mean[sorted_idx])
    # plt.xlabel("Feature Importance (XGB)")
    # plt.show()
    
    # fp_test_xgb, tp_test_xgb, threshold_test_xgb = roc_curve(pdv_1_vs_all_test, pred_test_xgb)
    # area_test_xgb = auc(fp_test_xgb,tp_test_xgb)
    
    
    
    # xyxyx = clf_logit_all.predict(df_4_nodrop)  
    
    # ccc=0
    # for k in pdvnd_1_vs_all.index:
    #     if (pdvnd_1_vs_all[pdvnd_1_vs_all.index[ccc]]==1 and pred_train_xgb[ccc]==0):
    #         df_fn_results = pd.concat((df_fn_results, df_nodrop[df_nodrop.index[ccc]].astype('Series')))
    #     if (pdvnd_1_vs_all[pdvnd_1_vs_all.index[ccc]]==0 and pred_train_xgb[ccc]==1):
    #         df_fp_results = pd.concat((df_fp_results, df_nodrop[df_nodrop.index[ccc]].astype('Series')))
    #     ccc+=1

    # ccc=0
    # for k in pdvnd_1_vs_all.index:
    #     if (pdvnd_1_vs_all[k]==1 and pred_train_xgb[ccc]==0):
    #         df_fn_results = pd.concat((df_fn_results, df_nodrop[k].astype('Series')))
    #     if (pdvnd_1_vs_all[k]==0 and pred_train_xgb[ccc]==1):
    #         df_fp_results = pd.concat((df_fp_results, df_nodrop[k].astype('Series')))
    #     ccc+=1    
    
    
    
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
    
    
    
    
    
    ###############logit
    ##############
    imb_weight = {
        0: 1,
        1: 1.11,
        }
    
    
    
    
    # clf_logit_all = sklm.LogisticRegression(C=2.0,penalty='l1', solver='saga', max_iter=3000, tol=1e-3, class_weight = 'balanced').fit(df_4, pdv_1_vs_all)
    # #print("all vs pd score = ", clf_logit_all.score(df_4,pdv_1_vs_all))
    # #class_weight = imb_weight
    
    # pred_nd_logit = clf_logit_all.predict(df_4_nodrop).copy()
    # df_diff_logit = pd.DataFrame()
    # loopcounter=0
    # for i in pdvnd_1_vs_all.index:
    #     if ((pdvnd_1_vs_all[i]!=1) and (pred_nd_logit[loopcounter]==1)):
    #         df_diff_logit = pd.concat((df_diff_logit, pd.Series([i])))
    #     loopcounter+=1
            
    # pred_nd_logit_test = clf_logit_all.predict(df_test_4_nodrop).copy()     
    # loopcounter=0  
    # df_diff_logit_test = pd.DataFrame()
    # for i in pdvnd_1_vs_all_test.index:
    #     if ((pdvnd_1_vs_all_test[i]!=1) and (pred_nd_logit_test[loopcounter]==1)):
    #         df_diff_logit = pd.concat((df_diff_logit, pd.Series([i])))
    #     loopcounter+=1
    
    # df_fp_results = pd.concat((df_fp_results, df_diff.xgb.copy()))
    
    # df_diff_xgb = pd.concat((df_fp_results.copy(), df_fn_results.copy()))
    
    
    
    # pred_proba_train_logit = clf_logit_all.predict_proba(df_4)[:,1].copy()
    # pred_proba_test_logit = clf_logit_all.predict_proba(df_test_4)[:,1].copy()
    
    # for i in range(0,len(pred_proba_train_logit)):
    #     if pred_proba_train_logit[i] >output_threshold:
    #         pred_proba_train_logit[i] = 1
    #     else:
    #         pred_proba_train_logit[i] = 0
            
    # for i in range(0,len(pred_proba_test_logit)):
    #     if pred_proba_test_logit[i] >output_threshold:
    #         pred_proba_test_logit[i] = 1
    #     else:
    #         pred_proba_test_logit[i] = 0
    
    # cm_logit_all = confusion_matrix(pdv_1_vs_all, pred_proba_train_logit)
    # sens_logit_all = cm_logit_all[1,1] / (cm_logit_all[1,1] + cm_logit_all[1,0])
    # spec_logit_all = cm_logit_all[0,0] / (cm_logit_all[0,0] + cm_logit_all[0,1])
    # prec_logit_all = cm_logit_all[1,1] / (cm_logit_all[1,1] + cm_logit_all[0,1])
    
    # cm_logit_all_nodrop = confusion_matrix(pdvnd_1_vs_all, pred_proba_train_logit)
    # sens_logit_all_nodrop = cm_logit_all_nodrop[1,1] / (cm_logit_all_nodrop[1,1] + cm_logit_all_nodrop[1,0])
    # spec_logit_all_nodrop = cm_logit_all_nodrop[0,0] / (cm_logit_all_nodrop[0,0] + cm_logit_all_nodrop[0,1])
    # prec_logit_all_nodrop = cm_logit_all_nodrop[1,1] / (cm_logit_all_nodrop[1,1] + cm_logit_all_nodrop[0,1])
    
    # cm_test_logit_all = confusion_matrix(pdv_1_vs_all_test, pred_proba_test_logit)
    # sens_test_logit_all = cm_test_logit_all[1,1] / (cm_test_logit_all[1,1] + cm_test_logit_all[1,0])
    # spec_test_logit_all = cm_test_logit_all[0,0] / (cm_test_logit_all[0,0] + cm_test_logit_all[0,1])
    # prec_test_logit_all = cm_test_logit_all[1,1] / (cm_test_logit_all[1,1] + cm_test_logit_all[0,1])
    
    # cm_test_logit_all_nodrop = confusion_matrix(pdvnd_1_vs_all_test, pred_proba_test_logit)
    # sens_test_logit_all_nodrop = cm_test_logit_all_nodrop[1,1] / (cm_test_logit_all_nodrop[1,1] + cm_test_logit_all_nodrop[1,0])
    # spec_test_logit_all_nodrop = cm_test_logit_all_nodrop[0,0] / (cm_test_logit_all_nodrop[0,0] + cm_test_logit_all_nodrop[0,1])
    # prec_test_logit_all_nodrop = cm_test_logit_all_nodrop[1,1] / (cm_test_logit_all_nodrop[1,1] + cm_test_logit_all_nodrop[0,1])
    
    
    
    # # cm_logit_all = confusion_matrix(pdv_1_vs_all, clf_logit_all.predict(df_4))
    # # sens_logit_all = cm_logit_all[1,1] / (cm_logit_all[1,1] + cm_logit_all[1,0])
    # # spec_logit_all = cm_logit_all[0,0] / (cm_logit_all[0,0] + cm_logit_all[0,1])
    # # prec_logit_all = cm_logit_all[1,1] / (cm_logit_all[1,1] + cm_logit_all[0,1])
    
    # # cm_logit_all_nodrop = confusion_matrix(pdvnd_1_vs_all, clf_logit_all.predict(df_4_nodrop))
    # # sens_logit_all_nodrop = cm_logit_all_nodrop[1,1] / (cm_logit_all_nodrop[1,1] + cm_logit_all_nodrop[1,0])
    # # spec_logit_all_nodrop = cm_logit_all_nodrop[0,0] / (cm_logit_all_nodrop[0,0] + cm_logit_all_nodrop[0,1])
    # # prec_logit_all_nodrop = cm_logit_all_nodrop[1,1] / (cm_logit_all_nodrop[1,1] + cm_logit_all_nodrop[0,1])
    
    # # cm_test_logit_all = confusion_matrix(pdv_1_vs_all_test, clf_logit_all.predict(df_test_4))
    # # sens_test_logit_all = cm_test_logit_all[1,1] / (cm_test_logit_all[1,1] + cm_test_logit_all[1,0])
    # # spec_test_logit_all = cm_test_logit_all[0,0] / (cm_test_logit_all[0,0] + cm_test_logit_all[0,1])
    # # prec_test_logit_all = cm_test_logit_all[1,1] / (cm_test_logit_all[1,1] + cm_test_logit_all[0,1])
    
    # # cm_test_logit_all_nodrop = confusion_matrix(pdvnd_1_vs_all_test, clf_logit_all.predict(df_test_4_nodrop))
    # # sens_test_logit_all_nodrop = cm_test_logit_all_nodrop[1,1] / (cm_test_logit_all_nodrop[1,1] + cm_test_logit_all_nodrop[1,0])
    # # spec_test_logit_all_nodrop = cm_test_logit_all_nodrop[0,0] / (cm_test_logit_all_nodrop[0,0] + cm_test_logit_all_nodrop[0,1])
    # # prec_test_logit_all_nodrop = cm_test_logit_all_nodrop[1,1] / (cm_test_logit_all_nodrop[1,1] + cm_test_logit_all_nodrop[0,1])
    
    # # ccc=0
    # # for k in pdvnd_1_vs_all.index:
    # #     if (pdvnd_1_vs_all[pdvnd_1_vs_all.index[ccc]]==1 and clf_logit_all.predict(df_4_nodrop)[ccc]==0):
    # #         df_fn_results = pd.concat((df_fn_results, df_nodrop[df_nodrop.index[ccc]].astype('Series')))
    # #     if (pdvnd_1_vs_all[pdvnd_1_vs_all.index[ccc]]==0 and clf_logit_all.predict(df_4_nodrop)[ccc]==1):
    # #         df_fp_results = pd.concat((df_fp_results, df_nodrop[df_nodrop.index[ccc]].astype('Series')))
    # #     ccc+=1
    # xyxyx = clf_logit_all.predict(df_4_nodrop)    
    # # ccc=0
    # # for k in pdvnd_1_vs_all.index:
    # #     if (pdvnd_1_vs_all[k]==1 and clf_logit_all.predict(df_4_nodrop)[ccc]==0):
    # #         df_fn_results = pd.concat((df_fn_results, df_nodrop[k].astype('Series')))
    # #     if (pdvnd_1_vs_all[k]==0 and clf_logit_all.predict(df_4_nodrop)[ccc]==1):
    # #         df_fp_results = pd.concat((df_fp_results, df_nodrop[k].astype('Series')))
    # #     ccc+=1    
    # #print("acc_frame_time logit sens, spec, prec =",sens_logit_all,",", spec_logit_all,",", prec_logit_all)

    
    # fp_logit_all, tp_logit_all, threshold_logit_all = roc_curve(pdv_1_vs_all, pred_proba_train_logit)
    # area_logit_all = auc(fp_logit_all,tp_logit_all)
    
    # logit_all_result= pd.DataFrame(data=[["logit all", sens_logit_all,spec_logit_all,prec_logit_all, area_logit_all]])
    
    # df_ML_results = pd.concat((df_ML_results, logit_all_result), ignore_index=True)

    # # print(clf_logit_all.coef_)
    # # plt.figure(11)
    # # perm_importance_logit = permutation_importance(clf_logit_all, df_test_4, pdv_1_vs_all_test)

    # # sorted_idx = perm_importance_logit.importances_mean.argsort()
    # # plt.barh(features[sorted_idx], perm_importance_logit.importances_mean[sorted_idx])
    # # plt.xlabel("Feature Importance (Logit)")
    # # plt.show()
    
    # fp_test_logit_all, tp_test_logit_all, threshold_test_logit_all = roc_curve(pdv_1_vs_all_test, pred_proba_test_logit)
    # area_test_logit_all = auc(fp_test_logit_all,tp_test_logit_all)
    
    # logit_all_result_test= pd.DataFrame(data=[["logit test all", sens_test_logit_all,spec_test_logit_all,prec_test_logit_all, area_test_logit_all]])
    
    # df_ML_results = pd.concat((df_ML_results, logit_all_result_test), ignore_index=True)
    ##########################
    ##############logit
    
    
    
    
    
    
    
    
    ##############nn
    #########
    # df_4_pd = pd.DataFrame(data=df_4)
    # df_4_train_balanced = df_4_pd.sample(frac = len(df_test_4)/len(df_4))
    
    # df_4_pd0 = df[df['PD Validity']==0]
    # df_4_pd1 = df[df['PD Validity']==1]
    # df_4_pd2 = df[df['PD Validity']==2]
    # df_4_pd0 = pd.concat((df_4_pd0,df_4_pd2))
    # df_4_pd0 = df_4_pd0.sample(frac=scale_imbalanced_class)
    # df_4_nn = pd.concat((df_4_pd0,df_4_pd1))
    
    
    
    # # df_test_4_pd = pd.DataFrame(data=df_test_4)
    # # df_test_4_train_balanced = df_test_4_pd.sample(frac = len(df_test_4)/len(df_test_4))
    # df_test_4_pd0 = df_test[df_test['PD Validity']==0]
    # df_test_4_pd1 = df_test[df_test['PD Validity']==1]
    # df_test_4_pd2 = df_test[df_test['PD Validity']==2]
    # df_test_4_pd0 = pd.concat((df_test_4_pd0,df_test_4_pd2))
    # df_test_4_pd0 = df_test_4_pd0.sample(frac=scale_imbalanced_class)
    # df_test_4_nn = pd.concat((df_test_4_pd0,df_test_4_pd1))
    
    # pdv_nn = df_4_nn['PD Validity'].reset_index(drop=True).copy()
    # pdvnd_nn = df_4_nn['PD Validity']
    # pdv_nn_test = df_test_4_nn['PD Validity'].reset_index(drop=True).copy() 
    # pdvnd_nn_test = df_test_4_nn['PD Validity']
    
    # df_4_nn = df_4_nn.drop(columns=df_4_nn.iloc[:,0:4])
    # df_4_nn_nd = df_4_nn.copy()
    # df_4_nn = df_4_nn.reset_index(drop=True)
    # df_test_4_nn = df_test_4_nn.drop(columns=df_test_4_nn.iloc[:,0:4])
    # df_test_4_nn_nd = df_test_4_nn.copy()
    # df_test_4_nn = df_test_4_nn.reset_index(drop=True)


    # pdv_nn_1_vs_all = pdv_nn.copy()
    # pdv_nn_0_vs_all = pdv_nn.copy()
    # pdvnd_nn_1_vs_all = pdvnd_nn.copy()
    # pdvnd_nn_0_vs_all = pdvnd_nn.copy()
    # pdv_nn_1_vs_all_test = pdv_nn_test.copy()
    # pdv_nn_0_vs_all_test = pdv_nn_test.copy()
    # pdvnd_nn_1_vs_all_test = pdvnd_nn_test.copy()
    # pdvnd_nn_0_vs_all_test = pdvnd_nn_test.copy()

    # for i in range (0,len(pdv_nn)):
    #     if pdv_nn[i]==2:
    #         pdv_nn_1_vs_all[i] = 0
    #         pdv_nn_0_vs_all[i] = 1

    # for i in range (0,len(pdv_nn_test)):
    #     if pdv_nn_test[i]==2:
    #         pdv_nn_1_vs_all_test[i] = 0
    #         pdv_nn_0_vs_all_test[i] = 1

    # for i in pdvnd_nn.index:
    #     if pdvnd_nn[i]==2:
    #         pdvnd_nn_1_vs_all[i] = 0
    #         pdvnd_nn_0_vs_all[i] = 1

    # for i in pdvnd_nn_test.index:
    #     if pdvnd_nn_test[i]==2:
    #         pdvnd_nn_1_vs_all_test[i] = 0
    #         pdvnd_nn_0_vs_all_test[i] = 1
            
    
    # clf_nn_all = MLPClassifier(activation='relu',hidden_layer_sizes=(700,300), max_iter=4000, alpha=0.02, learning_rate_init=0.001, solver='adam', tol=1e-3).fit(df_4_nn, pdv_nn_1_vs_all)
    # #,early_stopping=True
    # #clf_nn_all = SelectFromModel(clf_nn_all,threshold="0.5*mean",prefit=False, importance_getter='roc_auc').fit_transform(df_4_nn, pdv_nn_1_vs_all)#, threshold="200*mean", max_features=10, importance_getter='roc_auc', prefit=True
    

    # pred_proba_train_nn = clf_nn_all.predict_proba(df_4_nn)[:,1].copy()
    # pred_proba_test_nn = clf_nn_all.predict_proba(df_test_4_nn)[:,1].copy()
    
    # for i in range(0,len(pred_proba_train_nn)):
    #     if pred_proba_train_nn[i] >output_threshold:
    #         pred_proba_train_nn[i] = 1
    #     else:
    #         pred_proba_train_nn[i] = 0
            
    # for i in range(0,len(pred_proba_test_nn)):
    #     if pred_proba_test_nn[i] >output_threshold:
    #         pred_proba_test_nn[i] = 1
    #     else:
    #         pred_proba_test_nn[i] = 0
    
    # # loopcounter=0
    # # for i in pdvnd_nn_1_vs_all.index:
    # #     if ((pdvnd_nn_1_vs_all[i]!=1) and (pred_nd_nn[loopcounter]==1)):
    # #         df_diff_nn = pd.concat((df_diff_nn, pd.Series([i])))
    # #     loopcounter+=1
            
    # # pred_nd_nn_test = clf_nn_all.predict(df_test_4_nn_nodrop)     
    # # loopcounter=0  
    # # df_diff_nn = pd.DataFrame()
    # # for i in pdvnd_nn_1_vs_all_test.index:
    # #     if ((pdvnd_nn_1_vs_all_test[i]!=1) and (pred_nd_nn_test[loopcounter]==1)):
    # #         df_diff_nn = pd.concat((df_diff_nn, pd.Series([i])))
    # #     loopcounter+=1
   
    
    # pred_nd_nn = clf_nn_all.predict(df_4_nn_nd).copy()
    # df_diff_nn = pd.DataFrame()
    
    # loopcounter=0
    # for i in pdvnd_nn_1_vs_all.index:
    #     # if ((pdvnd_nn_1_vs_all[i]!=1) and (pred_nd_nn[loopcounter]==1)):
    #     if (pdvnd_nn_1_vs_all[i] != pred_nd_nn[loopcounter]):
    #         df_diff_nn = pd.concat((df_diff_nn, pd.Series([i])))
    #     loopcounter+=1
            
    # pred_nd_nn_test = clf_nn_all.predict(df_test_4_nn_nd).copy()
    # loopcounter=0
    # df_diff_nn = pd.DataFrame()
    # for i in pdvnd_nn_1_vs_all_test.index:
    #     # if ((pdvnd_nn_1_vs_all_test[i]!=1) and (pred_nd_nn_test[loopcounter]==1)):
    #     if (pdvnd_nn_1_vs_all_test[i] != pred_nd_nn_test[loopcounter]):
    #         df_diff_nn = pd.concat((df_diff_nn, pd.Series([i])))
    #     loopcounter+=1
        
    
    # cm_nn_all = confusion_matrix(pdv_nn_1_vs_all, pred_proba_train_nn)
    # cm_test_nn_all = confusion_matrix(pdv_nn_1_vs_all_test, pred_proba_test_nn)
    
    
    # sens_nn_all = cm_nn_all[1,1] / (cm_nn_all[1,1] + cm_nn_all[1,0])
    # spec_nn_all = cm_nn_all[0,0] / (cm_nn_all[0,0] + cm_nn_all[0,1])
    # prec_nn_all = cm_nn_all[1,1] / (cm_nn_all[1,1] + cm_nn_all[0,1])
    # sens_test_nn_all = cm_test_nn_all[1,1] / (cm_test_nn_all[1,1] + cm_test_nn_all[1,0])
    # spec_test_nn_all = cm_test_nn_all[0,0] / (cm_test_nn_all[0,0] + cm_test_nn_all[0,1])
    # prec_test_nn_all = cm_test_nn_all[1,1] / (cm_test_nn_all[1,1] + cm_test_nn_all[0,1])
    # #print("all neural net sens, spec, prec =",sens_nn_all,",", spec_nn_all,",", prec_nn_all)
    
    
    # fp_all, tp_all, threshold_nn_all = roc_curve(pdv_nn_1_vs_all, pred_proba_train_nn)
    # area_nn_all = auc(fp_all,tp_all)
    
    # nn_all_result= pd.DataFrame(data=[["nn all", sens_nn_all,spec_nn_all,prec_nn_all, area_nn_all]])
    
    # df_ML_results = pd.concat((df_ML_results, nn_all_result), ignore_index=True)
    
    
    # fp_test_all, tp_test_all, threshold_test_nn_all = roc_curve(pdv_nn_1_vs_all_test, pred_proba_test_nn)
    # area_test_nn_all = auc(fp_test_all,tp_test_all)
    
    # nn_all_result_test= pd.DataFrame(data=[["nn test all", sens_test_nn_all,spec_test_nn_all,prec_test_nn_all, area_test_nn_all]])
    
    # df_ML_results = pd.concat((df_ML_results, nn_all_result_test), ignore_index=True)
    
    
    
    
    
    
    # plt.figure(9)
    # perm_importance_nn = permutation_importance(clf_nn_all, df_test_4, pdv_nn_1_vs_all_test)

    # sorted_idx = perm_importance_nn.importances_mean.argsort()
    # plt.barh(features[sorted_idx], perm_importance_nn.importances_mean[sorted_idx])
    # plt.xlabel("Permutation Importance (NN)")
    # plt.show()
    

    #######################
    ##############nn
    
    
    
    
    
    
    
    #########svc
    ####################
    
    # clf_svc_all = SVC(C=1, kernel='poly', tol=1e-3, max_iter=3000, probability=True).fit(df_4, pdv_1_vs_all)
    
    # pred_nd_svc = clf_svc_all.predict(df_4_nodrop)
    # df_diff_svc = pd.DataFrame()
    # loopcounter=0
    # for i in pdvnd_1_vs_all.index:
    #     if ((pdvnd_1_vs_all[i]!=1) and (pred_nd_svc[loopcounter]==1)):
    #         df_diff_svc = pd.concat((df_diff_svc, pd.Series([i])))
    #     loopcounter+=1
            
    # pred_nd_svc_test = clf_svc_all.predict(df_test_4_nodrop)     
    # loopcounter=0  
    # df_diff_svc = pd.DataFrame()
    # for i in pdvnd_1_vs_all_test.index:
    #     if ((pdvnd_1_vs_all_test[i]!=1) and (pred_nd_svc_test[loopcounter]==1)):
    #         df_diff_svc = pd.concat((df_diff_svc, pd.Series([i])))
    #     loopcounter+=1
    
    # cm_svc_all = confusion_matrix(pdv_1_vs_all, clf_svc_all.predict(df_4))
    # sens_svc_all = cm_svc_all[1,1] / (cm_svc_all[1,1] + cm_svc_all[1,0])
    # spec_svc_all = cm_svc_all[0,0] / (cm_svc_all[0,0] + cm_svc_all[0,1])
    # prec_svc_all = cm_svc_all[1,1] / (cm_svc_all[1,1] + cm_svc_all[0,1])
    
    # cm_test_svc_all = confusion_matrix(pdv_1_vs_all_test, clf_svc_all.predict(df_test_4))
    # sens_test_svc_all = cm_test_svc_all[1,1] / (cm_test_svc_all[1,1] + cm_test_svc_all[1,0])
    # spec_test_svc_all = cm_test_svc_all[0,0] / (cm_test_svc_all[0,0] + cm_test_svc_all[0,1])
    # prec_test_svc_all = cm_test_svc_all[1,1] / (cm_test_svc_all[1,1] + cm_test_svc_all[0,1])
    

    # fp_svc_all, tp_svc_all, threshold_svc_all = roc_curve(pdv_1_vs_all, clf_svc_all.predict_proba(df_4)[:,1])
    # area_svc_all = auc(fp_svc_all,tp_svc_all)
    
    # svc_all_result= pd.DataFrame(data=[["svc all", sens_svc_all,spec_svc_all,prec_svc_all, area_svc_all]])
    
    # fp_test_svc_all, tp_test_svc_all, threshold_test_svc_all = roc_curve(pdv_1_vs_all_test, clf_svc_all.predict_proba(all_test_reshaped_scaled)[:,1])
    # area_test_svc_all = auc(fp_test_svc_all,tp_test_svc_all)
    
    # svc_all_result_test= pd.DataFrame(data=[["svc test all", sens_test_svc_all,spec_test_svc_all,prec_test_svc_all, area_test_svc_all]])
    
    # plt.figure(10)
    # perm_importance_svc = permutation_importance(clf_svc_all, df_test_4, pdv_1_vs_all_test)

    # sorted_idx = perm_importance_svc.importances_mean.argsort()
    # plt.barh(features[sorted_idx], perm_importance_svc.importances_mean[sorted_idx])
    # plt.xlabel("Permutation Importance (SVC)")
    # plt.show()
    
    # df_ML_results = pd.concat((df_ML_results, svc_all_result), axis=0,ignore_index=True)
    
    # df_ML_results = pd.concat((df_ML_results, svc_all_result_test), axis=0,ignore_index=True)
    
    
    
    
    ##########plot
    ################
    
#     plt.figure(7)
#     plt.plot(fp_all,tp_all, 'r-', label ='NN, AUC = %.3f'%area_nn_all)
#     #plt.plot(fp_svc_all,tp_svc_all, 'g-', label ='SVC, AUC = %.3f'%area_svc_all)
#     plt.plot(fp_logit_all,tp_logit_all, 'b-', label ='Logit, AUC = %.3f'%area_logit_all)
#     plt.plot(fp_xgb,tp_xgb, 'g-', label ='XGB, AUC = %.3f'%area_xgb)
    
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Comparison of all ML technique\'s ROC, Training')
#     plt.legend()
#     plt.show()
# #     

#     plt.figure(8)
#     plt.plot(fp_test_all,tp_test_all, 'r-', label ='NN, AUC = %.3f'%area_test_nn_all)
#     #plt.plot(fp_test_svc_all,tp_test_svc_all, 'g-', label ='SVC, AUC = %.3f'%area_test_svc_all)
#     plt.plot(fp_test_logit_all,tp_test_logit_all, 'b-', label ='Logit, AUC = %.3f'%area_test_logit_all)
#     plt.plot(fp_test_xgb,tp_test_xgb, 'g-', label ='XGB, AUC = %.3f'%area_test_xgb)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Comparison of all ML technique\'s ROC, Test')
#     plt.legend()
#     plt.show()

# =============================================================================
    counter_main_loop+=1
    
    print("main loop is on pass {} out of {}".format(counter_main_loop, trycount), end='\r')
    time.sleep(1)



df_ML_results = df_ML_results.rename(columns={0:'Features',1:'Sens', 2:'Spec', 3:'Prec', 4:'ROC AUC'})    