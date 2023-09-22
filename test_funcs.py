import datetime
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import shap
from pandas.core.common import SettingWithCopyWarning
import functions_for_classifier as funcs
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.3f}".format
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', '.*do not.*')

DATETIMEFORMAT = '%Y-%m-%d %H:%M:%S'
PATIENT_GROUPING = True
SAMPLEFRAC = 1.0
TRYCOUNT = 1
BUCKETS_COUNT = 300
THRESHOLD_BINS_COUNT = 300
UNDERSAMPLE = 0
JUMBLE_DATA = 0
SCALE_IMBALANCED_CLASS = 1
PLOT_DAILY_HIST_FOR_MIN_DAILY = False
NO_OF_WATERFALL_PLOTS = 5

EXTRACTED_COLUMN_LIST = [
    'userId',
    'dateTime',
    'point_awake',
    'point_sleep',
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
                 input_data_path,
                 input_data_dir,
                 model_path,
                 output_threshold=0.6,
                 importance_plot=False,
                 plot_xgb_roc=False,
                 override=True,
                 split_patient_output=False,
                 verbosity=1,
                 limit_floor_thr=0.45,
                 step_size=0.05,
                 output_dir='.',
                 output_filename=None,
                 translate=False,
                 min_frames_daily=20,
                 print_proba=False,
                 ):

        self.input_data_path = input_data_path
        self.input_data_dir = input_data_dir
        self.output_threshold = output_threshold
        self.model_path = model_path
        self.importance_plot = importance_plot
        self.plot_xgb_roc = plot_xgb_roc
        self.override = override
        self.split_patient_output = split_patient_output
        self.verbosity = verbosity
        self.limit_floor_thr = limit_floor_thr
        self.step_size = step_size
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.translate = translate
        self.min_frames_daily = min_frames_daily
        self.print_proba = print_proba

        self.test_data = None
        self.input_data = None

        if self.verbosity >= 2:
            print('initializing completed')

    def process_test(self):
        start_time = datetime.datetime.now()

        if self.verbosity >= 1:
            print("model used =", self.model_path)
            print("output_threshold = {}".format(self.output_threshold))

        df_input = funcs.load_data(input_data=self.input_data_path,
                                   path_data=self.input_data_dir).reset_index()

        test_data = df_input.copy()

        test_data = test_data[EXTRACTED_COLUMN_LIST + ['index']]
        test_data.rename(columns={'index': 'index_orig'}, inplace=True)

        test_data['activity_level'] = ((test_data['point_awake']) /
                                       (test_data['point_sleep'] + test_data['point_awake']))
        if self.verbosity >= 2:
            print('data length before na filter =', test_data.shape[0])
        test_data = test_data.dropna(subset=FEATURES_LIST)

        test_data = test_data[test_data['dashboardMode'] == 'RR']
        test_data_nd = test_data.copy()
        test_data = test_data.drop(columns=[
            'dashboardMode',
            'userId',
            'dateTime',
            'point_awake',
            'point_sleep',
        ])
        if self.verbosity >= 1:
            print('clean test data length =', test_data.shape[0])

        test_data['skin_temperature'] = funcs.scale_skin_temp(test_data['skin_temperature'])
        test_data['val_sd_signal_w_sqa'] = funcs.val_sd_scaler(test_data['val_sd_signal_w_sqa'])

        if self.translate:
            feature_language_dict = {}
            for i in range(0, len(FEATURES_LIST)):
                feature_language_dict[FEATURES_LIST[i]] = IRL_FEATURE_NAMES[i]
            test_data.rename(columns=feature_language_dict, inplace=True)
            test_data_nd.rename(columns=feature_language_dict, inplace=True)

        self.test_data = test_data
        self.input_data = df_input
        self.test_data_nd = test_data_nd

        stop_time = datetime.datetime.now()
        if self.verbosity >= 2:
            print(
                f'Data preprocessing is done, duration = {(stop_time - start_time).total_seconds():.3} seconds')

        print('preprocessing done')

    def predict_data(self):
        start_time = datetime.datetime.now()

        if self.translate:
            dtest = xgb.DMatrix(
                self.test_data[self.test_data.columns.difference(['index_orig'])][IRL_FEATURE_NAMES])
        else:
            dtest = xgb.DMatrix(
                self.test_data[self.test_data.columns.difference(['index_orig'])][FEATURES_LIST])

        bst = xgb.Booster({'nthread': 4})
        bst.load_model(self.model_path)

        if self.translate:
            bst.feature_names = IRL_FEATURE_NAMES
        else:
            bst.feature_names = FEATURES_LIST

        if self.importance_plot:
            funcs.plot_importance(self, bst, NO_OF_WATERFALL_PLOTS)

        if self.verbosity >= 2:
            print('prediction in progress...')

        pred_test_xgb_raw = bst.predict(dtest)
        pred_test_xgb_raw_df = pd.DataFrame(data=pred_test_xgb_raw, columns=[
                                            'pred_proba'], index=self.test_data['index_orig'])

        pred_test_xgb = pred_test_xgb_raw.copy()
        if self.verbosity >= 2:
            print('done predicting, writing now...')

        pred_test_xgb[pred_test_xgb > self.output_threshold] = 1
        pred_test_xgb[pred_test_xgb <= self.output_threshold] = 0
        pred_test_xgb = [int(v) if not np.isnan(v) else np.nan for v in pred_test_xgb]

        self.test_data['pdv_pred'] = pred_test_xgb

        self.input_data = self.input_data.merge(self.test_data[['index_orig', 'pdv_pred']],
                                                how='left',
                                                left_on='index',
                                                right_on='index_orig')

        if self.override:
            self.input_data['PD Validity'] = self.input_data.apply(
                lambda row: row['pdv_pred'] if not np.isnan(row['pdv_pred']) else row['PD Validity'], axis=1)

        today = datetime.date.today()

        if self.min_frames_daily:
            self.input_data, self.test_data_nd = funcs.execute_min_frames_daily(
                self, pred_test_xgb, pred_test_xgb_raw_df, PLOT_DAILY_HIST_FOR_MIN_DAILY)

        if self.print_proba:
            if self.min_frames_daily is None:
                self.input_data = self.input_data.merge(pred_test_xgb_raw_df['pred_proba'],
                                                        how='left',
                                                        left_on='index',
                                                        right_on='index_orig')

        self.input_data = self.input_data.drop(columns=['index'])

        if self.split_patient_output:
            for pt in pd.unique(self.input_data['userId']):
                df_out = self.input_data[self.input_data['userId'] == pt]
                filename = f'{pt}_{today}_predicted_{"override" if self.override else "no_override"}.csv' if self.output_filename is None else f'{pt}_{self.output_filename}'
                df_out.to_csv(os.path.join(self.output_dir, filename), index=False)
                print(f'file outputted as {filename}')
        else:
            filename = f'{today}_predicted_{"override" if self.override else "no_override"}.csv' if self.output_filename is None else self.output_filename

            self.input_data.to_csv(os.path.join(self.output_dir, filename), index=False)
            print(f'file outputted as {filename}')

        stop_time = datetime.datetime.now()
        if self.verbosity >= 2:
            print(
                f'Data preprocessing is done, duration = {(stop_time - start_time).total_seconds():.3} seconds')
