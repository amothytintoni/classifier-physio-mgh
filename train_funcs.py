import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc
from sklearn.preprocessing import MinMaxScaler
import functions_for_classifier as funcs
import warnings
from pandas.core.common import SettingWithCopyWarning

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.3f}".format
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings('ignore', '.*do not.*')
plt.rcParams["figure.autolayout"] = True

DATETIMEFORMAT = '%Y-%m-%d %H:%M:%S'
DATETIMEFORMATFILESAVE = '%Y-%m-%d %H_%M_%S'
PATIENT_GROUPING = True
SAMPLEFRAC = 1.0
TRYCOUNT = 1
BUCKETS_COUNT = 300
THRESHOLD_BINS_COUNT = 300
UNDERSAMPLE = 0
JUMBLE_DATA = 0
SCALE_IMBALANCED_CLASS = 1
PLOT_XGB_ROC = False
OUTPUT_DFCM = False
NO_OF_WATERFALL_PLOTS = 5

EXTRACTED_COLUMN_LIST = [
    'userId',
    'dateTime',
    'PD Validity',
    'point_awake', 'point_sleep',
    'SQA_index',
    'accepted_frame_ratio',
    'list_sensor_onskin_status_ratio',
    'val_sd_signal_w_sqa',
    'list_onskin_status_stdev_ratio',
    'skin_temperature',
    'dashboardMode',
]

FEATURES_LIST = [
    'SQA_index',
    'accepted_frame_ratio',
    'list_sensor_onskin_status_ratio',
    'val_sd_signal_w_sqa',
    'list_onskin_status_stdev_ratio',
    'skin_temperature',
    'activity_level',
]

IRL_FEATURE_NAMES = [
    'Median Signal Quality Index',
    'Frequency and Sinusoidality Index',
    'Ratio of On-skin Period',
    'Amplitude Index',
    'Consistency of On-skin Status',
    'Skin Temperature',
    'Activity Level'
]


class Classifier_xgb:
    def __init__(self,
                 input_train='empty',
                 path_trainset='empty',
                 input_test='empty',
                 path_testset='empty',
                 output_threshold=0.6,
                 split_patient_output=False,
                 override=True,
                 print_diff_to_file=False,
                 print_score_to_file=False,
                 print_predict_result=False,
                 max_depth=7,
                 reg_value=30,
                 eta=0.1,
                 gamma=1,
                 reg_mode='l1',
                 eval_metric='auc',
                 scale_pos_weight=1,
                 num_round=150,
                 save_model_name='newest.model',
                 importance_plot=False,
                 min_child_weight=1,
                 verbosity=1,
                 translate=False,
                 output_dir='.',
                 ):
        self.input_train = input_train
        self.path_trainset = path_trainset
        self.input_test = input_test
        self.path_testset = path_testset
        self.output_threshold = output_threshold
        self.override = override
        self.split_patient_output = split_patient_output
        self.print_diff_to_file = print_diff_to_file
        self.print_score_to_file = print_score_to_file
        self.max_depth = max_depth
        self.reg_value = reg_value
        self.eta = eta
        self.gamma = gamma
        self.reg_mode = reg_mode
        self.eval_metric = eval_metric
        self.scale_pos_weight = scale_pos_weight
        self.num_round = num_round
        self.save_model_name = save_model_name
        self.print_predict_result = print_predict_result
        self.importance_plot = importance_plot
        self.min_child_weight = min_child_weight
        self.verbosity = verbosity
        self.translate = translate
        self.output_dir = output_dir
        if self.verbosity >= 2:
            print('initializing completed')

    def process_train(self):
        start_time = datetime.datetime.now()
        df_train = funcs.load_data(input_data=self.input_train, path_data=self.path_trainset)[
            EXTRACTED_COLUMN_LIST]
        df_test = funcs.load_data(input_data=self.input_test, path_data=self.path_testset)[
            EXTRACTED_COLUMN_LIST]
        df_input_ori = pd.concat((df_train, df_test), ignore_index=True)
        df_train_ori = df_train.copy()
        df_test_ori = df_test.copy()

        if self.verbosity >= 1:
            print("output_threshold = {}".format(self.output_threshold))

        df_train_nd = df_train.copy()
        df_test_nd = df_test.copy()

        df_train['activity_level'] = (
            (df_train['point_awake'])/(df_train['point_sleep']+df_train['point_awake']))
        df_train_nd['activity_level'] = (
            (df_train_nd['point_awake'])/(df_train_nd['point_sleep']+df_train_nd['point_awake']))

        df_test['activity_level'] = (
            (df_test['point_awake'])/(df_test['point_sleep']+df_test['point_awake']))
        df_test_nd['activity_level'] = (
            (df_test_nd['point_awake'])/(df_test_nd['point_sleep']+df_test_nd['point_awake']))

        df_train = df_train.dropna()
        df_train_nd = df_train_nd.dropna()
        df_test = df_test.dropna()
        df_test_nd = df_test_nd.dropna()

        if self.verbosity >= 2:
            print(
                f'train length after filtering = {df_train.shape[0]}\ntest length after filtering {df_test.shape[0]}')

        df_train = df_train[df_train['dashboardMode'] == 'RR']
        df_train = df_train.drop(columns=['dashboardMode'])
        df_train_nd = df_train_nd[df_train_nd['dashboardMode'] == 'RR']
        df_train_nd = df_train_nd.drop(columns=['dashboardMode'])

        df_test = df_test[df_test['dashboardMode'] == 'RR']
        df_test = df_test.drop(columns=['dashboardMode'])
        df_test_nd = df_test_nd[df_test_nd['dashboardMode'] == 'RR']
        df_test_nd = df_test_nd.drop(columns=['dashboardMode'])

        if UNDERSAMPLE == 1:
            df_test_pd0 = df_test[df_test['PD Validity'] == 0]
            df_test_pd1 = df_test[df_test['PD Validity'] == 1]
            df_test_pd2 = df_test[df_test['PD Validity'] == 2]
            df_test_pd0 = pd.concat((df_test_pd0, df_test_pd2))
            if df_test_pd1.shape[0]/df_test_pd0.shape[0] <= 1:
                df_test_pd0 = df_test_pd0.sample(
                    frac=df_test_pd1.shape[0]/df_test_pd0.shape[0])
            elif df_test_pd1.shape[0]/df_test_pd0.shape[0] > 1:
                df_test_pd1 = df_test_pd1.sample(
                    frac=df_test_pd0.shape[0]/df_test_pd1.shape[0])

            df_test = pd.concat((df_test_pd0, df_test_pd1))

        if JUMBLE_DATA == 1:
            df_all = pd.concat((df_train, df_test))
            df_all['index_ori'] = df_all.index
            df_all.reset_index(drop=True, inplace=True)
            df_train, df_test = train_test_split(df_all, test_size=len(
                df_test)/(len(df_train)+len(df_test)), shuffle=True)
        else:
            df_train['index_ori'] = df_train.index
            df_test['index_ori'] = df_test.index

        pdv_train = df_train['PD Validity'].copy()
        pdv_train_1_vs_all = pdv_train.copy()

        pdv_test = df_test['PD Validity'].copy()
        pdv_test_1_vs_all = pdv_test.copy()

        pdv_train_1_vs_all = funcs.convert_to_1_vs_all(pdv_train)
        pdv_test_1_vs_all = funcs.convert_to_1_vs_all(pdv_test)

        df_train = df_train.drop(columns=['userId',
                                          'dateTime',
                                          'PD Validity',
                                          'point_awake',
                                          'point_sleep',
                                          'index_ori',
                                          ])

        df_train['skin_temperature'] = funcs.scale_skin_temp(df_train['skin_temperature'])
        df_train['val_sd_signal_w_sqa'] = funcs.val_sd_scaler(df_train['val_sd_signal_w_sqa'])

        df_test = df_test.drop(columns=['userId',
                                        'dateTime',
                                        'PD Validity',
                                        'point_awake',
                                        'point_sleep',
                                        'index_ori', ])

        df_test['skin_temperature'] = funcs.scale_skin_temp(df_test['skin_temperature'])
        df_test['val_sd_signal_w_sqa'] = funcs.val_sd_scaler(df_test['val_sd_signal_w_sqa'])

        if self.verbosity >= 1:
            print('train length =',
                  df_train.shape[0], 'test length =', df_test.shape[0])

        if self.translate:
            df_train.rename(inplace=True, columns={FEATURES_LIST[0]: IRL_FEATURE_NAMES[0], FEATURES_LIST[1]: IRL_FEATURE_NAMES[1], FEATURES_LIST[2]: IRL_FEATURE_NAMES[2],
                            FEATURES_LIST[3]: IRL_FEATURE_NAMES[3], FEATURES_LIST[4]: IRL_FEATURE_NAMES[4], FEATURES_LIST[5]: IRL_FEATURE_NAMES[5], FEATURES_LIST[6]: IRL_FEATURE_NAMES[6]})
            df_test.rename(inplace=True, columns={FEATURES_LIST[0]: IRL_FEATURE_NAMES[0], FEATURES_LIST[1]: IRL_FEATURE_NAMES[1], FEATURES_LIST[2]: IRL_FEATURE_NAMES[2],
                                                  FEATURES_LIST[3]: IRL_FEATURE_NAMES[3], FEATURES_LIST[4]: IRL_FEATURE_NAMES[4], FEATURES_LIST[5]: IRL_FEATURE_NAMES[5], FEATURES_LIST[6]: IRL_FEATURE_NAMES[6]})

        self.df_train = df_train
        self.df_train_nd = df_train_nd
        self.df_test = df_test
        self.df_test_nd = df_test_nd
        self.pdv_test = pdv_test_1_vs_all
        self.pdv_train = pdv_train_1_vs_all
        self.df_input_ori = df_input_ori
        self.df_train_ori = df_train_ori
        self.df_test_ori = df_test_ori

        stop_time = datetime.datetime.now()
        if self.verbosity >= 2:
            print(
                f'Data preprocessing is done, duration = {(stop_time - start_time).total_seconds():.3} seconds')

    def train_model(self):
        start_time = datetime.datetime.now()
        dtrain = xgb.DMatrix(self.df_train.iloc[:, 0:7],
                             label=self.pdv_train
                             )

        dtest = xgb.DMatrix(self.df_test.iloc[:, 0:7],
                            label=self.pdv_test
                            )

        xgb_param = {'max_depth': int(self.max_depth),
                     'eta': float(self.eta),
                     'gamma': float(self.gamma),
                     'scale_pos_weight': int(self.scale_pos_weight),
                     'min_child_weight': float(self.min_child_weight),
                     'objective': 'binary:logistic',
                     'verbosity': 0,
                     }
        xgb_param['nthread'] = 4
        xgb_param['eval_metric'] = [self.eval_metric]
        if self.reg_mode == 'l1':
            xgb_param['alpha'] = float(self.reg_value)
        elif self.reg_mode == 'l2':
            xgb_param['lambda'] = float(self.reg_value)
        else:
            raise Exception(
                'Error: regularization mode you entered does not exist.')

        bst = xgb.train(xgb_param,
                        dtrain,
                        num_boost_round=int(self.num_round),
                        )
        bst.save_model(self.save_model_name)
        if self.verbosity >= 2:
            print('model successfully trained')

        xgb_param.pop('objective')
        xgb_param.pop('nthread')
        xgb_param.pop('verbosity')
        if self.verbosity >= 1:
            print('parameters:', xgb_param)

        if self.verbosity >= 2:
            print('getting statistics now...')

        if self.importance_plot:
            explainer = shap.Explainer(bst)
            if self.translate:
                shap_values = explainer(self.df_train[IRL_FEATURE_NAMES])
            else:
                shap_values = explainer(self.df_train[FEATURES_LIST])

            if self.test_data.shape[0] < NO_OF_WATERFALL_PLOTS:
                print_list = [i for i in range(0, self.test_data.shape[0])]
            else:
                print_list = [i for i in range(0, NO_OF_WATERFALL_PLOTS)]

            for i in print_list:
                shap.plots.waterfall(shap_values[i], show=False)
                plt.savefig(f'shap_waterfall{i}.png')
                plt.clf()

            shap.plots.bar(shap_values, show=False)
            plt.savefig('shap_bar.png')
            plt.clf()
            if self.translate:
                shap.summary_plot(shap_values, self.df_train[IRL_FEATURE_NAMES], show=False)
                plt.savefig('shap_summary.png')
                plt.clf()
            else:
                shap.summary_plot(shap_values, self.df_train[FEATURES_LIST], show=False)
                plt.savefig('shap_summary.png')
                plt.clf()

            ax = xgb.plot_importance(bst, xlabel="Weights-based F score",
                                     importance_type='weight', title="Weights-based Feature Importance")
            ax.figure.savefig('Weights-based_Importance.png')
            ax = xgb.plot_importance(bst, xlabel="Gains-based F score",
                                     importance_type='gain', title="Gains-based Feature Importance")
            ax.figure.savefig('Gains-based_Importance.png')
            ax = xgb.plot_importance(bst, xlabel="Coverage-based F score",
                                     importance_type='cover', title="Coverage-based Feature Importance")
            ax.figure.savefig('Coverage-based_Importance.png')

            ax = xgb.plot_importance(bst, xlabel="Total Gains score",
                                     importance_type='total_gain', title="Total Gains of Features")
            ax.figure.savefig('Total-gains_Importance.png')
            ax = xgb.plot_importance(bst, xlabel="Total Coverages score",
                                     importance_type='total_cover', title="Total Coverages of Features")
            ax.figure.savefig('Total-coverage_Importance.png')

        pred_train_xgb_raw = bst.predict(dtrain)
        pred_test_xgb_raw = bst.predict(dtest)

        pred_train_xgb = pred_train_xgb_raw.copy()
        pred_test_xgb = pred_test_xgb_raw.copy()

        pred_nd_xgb = pred_train_xgb.copy()
        df_diff_xgb = pd.DataFrame()

        for i in range(0, len(pred_train_xgb)):
            if pred_train_xgb[i] > self.output_threshold:
                pred_train_xgb[i] = 1
            else:
                pred_train_xgb[i] = 0

        for i in range(0, len(pred_test_xgb)):
            if pred_test_xgb[i] > self.output_threshold:
                pred_test_xgb[i] = 1
            else:
                pred_test_xgb[i] = 0

        cm_xgb = confusion_matrix(self.pdv_train, pred_train_xgb)
        sens_xgb = cm_xgb[1, 1] / (cm_xgb[1, 1] + cm_xgb[1, 0])
        spec_xgb = cm_xgb[0, 0] / (cm_xgb[0, 0] + cm_xgb[0, 1])
        prec_xgb = cm_xgb[1, 1] / (cm_xgb[1, 1] + cm_xgb[0, 1])

        if OUTPUT_DFCM:
            cm_test_xgb = confusion_matrix(self.pdv_test, pred_test_xgb)
            dfcm = pd.DataFrame(cm_test_xgb).to_csv('dfcm.csv', index=False)

        sens_test_xgb = cm_test_xgb[1, 1] / \
            (cm_test_xgb[1, 1] + cm_test_xgb[1, 0])
        spec_test_xgb = cm_test_xgb[0, 0] / \
            (cm_test_xgb[0, 0] + cm_test_xgb[0, 1])
        prec_test_xgb = cm_test_xgb[1, 1] / \
            (cm_test_xgb[1, 1] + cm_test_xgb[0, 1])

        fp_xgb, tp_xgb, threshold_xgb = funcs.my_roc(
            self.pdv_train, pred_train_xgb_raw, THRESHOLD_BINS_COUNT)
        area_xgb = auc(fp_xgb, tp_xgb)

        xgb_result = pd.DataFrame(
            data=[["xgb train", sens_xgb, spec_xgb, prec_xgb, area_xgb]])

        df_ML_results = pd.DataFrame()
        df_ML_results = pd.concat(
            (df_ML_results, xgb_result), ignore_index=True)

        fp_test_xgb, tp_test_xgb, threshold_test_xgb = funcs.my_roc(
            self.pdv_test, pred_test_xgb_raw, THRESHOLD_BINS_COUNT)
        area_test_xgb = auc(fp_test_xgb, tp_test_xgb)

        xgb_result_test = pd.DataFrame(
            data=[["xgb test", sens_test_xgb, spec_test_xgb, prec_test_xgb, area_test_xgb]])

        df_ML_results = pd.concat(
            (df_ML_results, xgb_result_test), ignore_index=True)

        if self.print_diff_to_file:
            pred_nd_xgb = pred_train_xgb.copy()
            df_diff_xgb = pd.DataFrame()
            df_diff_xgb_train = pd.DataFrame()
            df_diff_xgb_test = pd.DataFrame()

            loopcounter = 0

            for i in self.pdv_train.index:
                if self.pdv_train[i] != pred_nd_xgb[loopcounter]:
                    df_diff_xgb = pd.concat((df_diff_xgb, pd.Series(data=[i])))
                loopcounter += 1

            df_diff_xgb_train = df_diff_xgb.copy()

            df_diff_xgb = pd.DataFrame()
            pred_nd_xgb_test = pred_test_xgb.copy()
            loopcounter = 0
            for i in self.pdv_test.index:
                if self.pdv_test[i] != pred_nd_xgb_test[loopcounter]:
                    df_diff_xgb = pd.concat((df_diff_xgb, pd.Series(data=[i])))
                loopcounter += 1

            df_diff_xgb_test = df_diff_xgb.copy()
            df_diff_xgb = pd.DataFrame()
            df_diff_xgb = pd.concat((df_diff_xgb_train, df_diff_xgb_test))

            df_diff_xgb = df_diff_xgb.rename(columns={0: 'label'})
            df_diff_xgb.reset_index(drop=True, inplace=True)

            self.df_diff_xgb = df_diff_xgb
            self.df_diff_xgb_train = df_diff_xgb_train
            self.df_diff_xgb_test = df_diff_xgb_test

        if PLOT_XGB_ROC:

            plt.figure(10)
            plt.plot(fp_test_xgb, tp_test_xgb, 'r-', label='Test, AUC = %.3f' % area_test_xgb)
            plt.title('XGBoost ROC')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.savefig('Model ROC.png')
            plt.show()

        df_ML_results = df_ML_results.rename(
            columns={0: 'Optimizer', 1: 'Sens', 2: 'Spec', 3: 'Prec', 4: 'ROC AUC'})

        self.df_ML_results = df_ML_results
        self.xgb_param = xgb_param
        self.pred_test_xgb = pred_test_xgb
        self.pred_train_xgb = pred_train_xgb

        if self.verbosity >= 2:
            print('statistics compiled. writing now...')

        if self.print_diff_to_file:
            df_diff_xgb.to_csv('diff_file_{}_{}.csv'.format(
                self.save_model_name, datetime.date.today()), index=False)
        if self.print_score_to_file:
            df_ML_results.to_csv('score_file_{}_{}.csv'.format(
                self.save_model_name, datetime.datetime.now().strftime(DATETIMEFORMATFILESAVE)), index=False)

        if self.print_predict_result:
            self.df_input_ori['pdv_pred'] = np.empty(
                self.df_input_ori.shape[0])
            self.df_input_ori['pdv_pred'] = np.nan
            self.df_train_ori['pdv_pred'] = np.empty(
                self.df_train_ori.shape[0])
            self.df_train_ori['pdv_pred'] = np.nan
            self.df_test_ori['pdv_pred'] = np.empty(self.df_test_ori.shape[0])
            self.df_test_ori['pdv_pred'] = np.nan

            pred_test_xgb = [int(a) for a in pred_test_xgb]
            pred_test_xgb_df = pd.DataFrame(
                data=pred_test_xgb, index=self.df_test_nd.index, columns=['pdv'])
            pred_test_xgb_df['index_ori'] = pred_test_xgb_df.index

            pred_train_xgb_df = pd.DataFrame(
                data=pred_train_xgb, index=self.df_train_nd.index, columns=['pdv'])
            pred_train_xgb_df['index_ori'] = pred_train_xgb_df.index

            for i in pred_test_xgb_df.index:
                self.df_test_ori['pdv_pred'][pred_test_xgb_df['index_ori']
                                             [i]] = pred_test_xgb_df['pdv'][i]
                if self.override:
                    self.df_test_ori['PD Validity'][pred_test_xgb_df['index_ori']
                                                    [i]] = pred_test_xgb_df['pdv'][i]

            for i in pred_train_xgb_df.index:
                self.df_train_ori['pdv_pred'][pred_train_xgb_df['index_ori']
                                              [i]] = pred_train_xgb_df['pdv'][i]
                if self.override:
                    self.df_train_ori['PD Validity'][pred_train_xgb_df['index_ori']
                                                     [i]] = pred_train_xgb_df['pdv'][i]

            df_output_ori = pd.concat((self.df_train_ori, self.df_test_ori))

            if self.split_patient_output:
                for pt in pd.unique(df_output_ori['userId']):
                    df_out = df_output_ori[df_output_ori['userId'] == pt]
                    if self.override:
                        df_out.to_csv('{}_{}_predicted_with_override.csv'.format(
                            pt, datetime.date.today()), index=False)
                    else:
                        df_out.to_csv('{}_{}_predicted_no_override.csv'.format(
                            pt, datetime.date.today()), index=False)

            else:
                df_output_ori.to_csv('{}_result.csv'.format(
                    datetime.date.today()), index=False)

        stop_time = datetime.datetime.now()
        if self.verbosity >= 2:
            print(
                f'writing completed. Thanks for using dawg. Training and writing duration = {(stop_time - start_time).total_seconds():.3} seconds')
