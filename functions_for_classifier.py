import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import boto3
import pandas as pd
import os
from boto3.dynamodb.conditions import Key, Attr
import math
from copy import deepcopy
import datetime as dt
from datetime import datetime
import shap
import xgboost as xgb

MIN_SKIN_TEMP = 20.5
MAX_SKIN_TEMP = 44.8
MIN_VAL_SD_W_SQA = 0
MAX_VAL_SD_W_SQA = 2125.33
DATETIMEFORMAT1 = '%Y-%m-%d %H:%M:%S'
DATETIMEFORMAT2 = '%d/%m/%Y %H:%M'
DATETIMEFORMAT3 = '%Y-%m-%d %H:%M'
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


def convert_string_to_datetime(data):
    try:
        dtm = data['dateTime'].apply(lambda x: datetime.strptime(x, DATETIMEFORMAT1))
        data['dateTime'] = dtm
        return data
    except:
        raise Exception("Please input a dataframe, and ensure string type for dateTime column")


def convert_datetime_to_string(data):
    try:
        stringed = data['dateTime'].apply(lambda x: x.strftime(DATETIMEFORMAT1))
        data['dateTime'] = stringed
        return data
    except:
        raise Exception("Please input a dataframe, and ensure dateTime type for dateTime column")


def proba_to_pred(proba, thresh):

    for i in range(0, len(proba)):
        if proba[i] == np.nan:
            continue
        if proba[i] > thresh:
            proba[i] = 1
        else:
            proba[i] = 0
    return proba


def make_daily_id(dateTime, DATETIMEFORMAT):

    return int(10000*(dateTime.year - 2022) + 100*dateTime.month + dateTime.day)


def daily_id_to_datetime(daily_id, DATETIMEFORMAT):
    return '{}-{}-{}'.format(math.floor(daily_id/10000), math.floor((daily_id % 10000)/100), daily_id % 100)


def count1s(arr):
    countof1s = 0
    for q in range(0, len(arr)):
        if arr[q] == 1:
            countof1s += 1
    return countof1s


def check_all_proba(arr, threshold):
    flag = 0
    for i in range(0, len(arr)):
        if arr[i] > threshold:
            flag = 1

            break
    return flag


def val_sd_scaler(data):
    data = data.apply(lambda x: ((x - MIN_VAL_SD_W_SQA) / (MAX_VAL_SD_W_SQA - MIN_VAL_SD_W_SQA)))
    return data


def generate_activity_level(data, drop=True):
    try:
        data['activity_level'] = data['point_awake'] / \
            (data['point_sleep'] + data['point_awake'])
        if drop is True:
            data.drop(columns=['point_awake', 'point_sleep'], inplace=True)
    except:
        raise Exception(
            'one of the required columns do not exist, or you did not pass in a dataFrame object')

    return data


def excel_time_to_std_time(obj):
    if not isinstance(obj, datetime):
        obj = datetime.strptime(obj, DATETIMEFORMAT2)
    time_obj = f'{obj.year}-{obj.month:02}-{obj.day:02} {obj.hour:02}:{obj.minute:02}'

    return datetime.strptime(time_obj, DATETIMEFORMAT3)


def find_seconds(obj, ref):
    '''
    Retrieves "seconds" from ref if obj does not contain seconds. Verify that ref and obj is the same data point prior to getting it.
Args:
    ref (str): a string representing datetime
    obj (str): a string representing datetime
Output:
 - ref (datetime) with seconds information
 - False if obj and ref is not a match
    '''
    if not isinstance(ref, datetime):
        try:
            ref = datetime.strptime(ref, DATETIMEFORMAT1)
        except:
            try:
                ref = datetime.strptime(obj, DATETIMEFORMAT3)
            except:
                try:
                    ref = datetime.strptime(obj, DATETIMEFORMAT2)
                except:
                    raise Exception('reference dateTime is not standardized')

    if not isinstance(obj, datetime):
        try:
            a = datetime.strptime(obj, DATETIMEFORMAT2)
        except:
            try:
                a = datetime.strptime(obj, DATETIMEFORMAT3)
            except:
                try:
                    a = datetime.strptime(obj, DATETIMEFORMAT1)
                except:
                    raise Exception('datetime format unknown')

        finally:
            obj = a

    if (obj.minute == ref.minute) and (obj.hour == ref.hour) and (obj.day == ref.day) and (obj.month == ref.month) and (obj.year == ref.year):
        return ref
    else:
        return False


def scale_skin_temp(temps):
    temps = list(temps)
    newtemp = deepcopy(temps)
    for i in range(0, len(temps)):
        newtemp[i] = (temps[i] - MIN_SKIN_TEMP) / \
            (MAX_SKIN_TEMP - MIN_SKIN_TEMP)
    return np.ravel(newtemp)


def reverse_scaled_skin_temp(temps):
    temps = list(temps)
    newtemp = deepcopy(temps)
    for i in range(0, len(temps)):
        newtemp[i] = temps[i]*(MAX_SKIN_TEMP - MIN_SKIN_TEMP) + MIN_SKIN_TEMP
    return np.ravel(newtemp)


def calculate_metrics(conf_matrix):
    assert len(conf_matrix) == 2
    assert len(conf_matrix[0]) == 2
    sens = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    spec = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    prec = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    return [sens, spec, prec]


def load_data(input_data, path_data):
    if input_data is None and path_data is None:
        raise Exception(
            'please enter input file or the folder which contains multiple files bro')
    if input_data is not None and path_data is None:
        df_test = pd.read_csv(input_data, low_memory=False)

    elif input_data is not None and path_data is not None:

        df_test = pd.read_csv(os.path.join(path_data, input_data), low_memory=False)

    elif input_data is None and path_data is not None:
        df_test = pd.DataFrame()

        list_file_test = [x for x in os.listdir(path_data) if x.endswith('.csv')]

        for testfile in list_file_test:
            read_data = pd.read_csv(os.path.join(path_data, testfile), low_memory=False)
            df_test = pd.concat((df_test, read_data), ignore_index=True)

    return df_test


def convert_to_1_vs_all(data):

    data_1_vs_all = data.copy()

    for i in data.index:

        if data[i] == 2:
            data_1_vs_all[i] = 0

    return data_1_vs_all


def my_roc(truth, pred_prob, threshold_bins):
    n = len(truth)
    truth_self = truth.copy()
    pred_prob_self = pred_prob.copy()
    pp_temp = pred_prob_self.copy()
    threshold = np.linspace(0, 1+1/threshold_bins, threshold_bins)
    cm = np.zeros((len(threshold), 2, 2))
    counter = 0
    for th in threshold:
        pp_temp = pred_prob_self.copy()
        for i in range(0, len(pp_temp)):
            if pp_temp[i] <= th:
                pp_temp[i] = 0
            else:
                pp_temp[i] = 1
        cm[counter] = confusion_matrix(truth_self, pp_temp, normalize='true')
        counter += 1

    fps = np.zeros(len(threshold))
    tps = np.zeros(len(threshold))

    for i in range(0, len(threshold)):
        fps[i] = cm[i][0, 1]
        tps[i] = cm[i][1, 1]

    return fps, tps, threshold


def plot_importance(clf, bst, no_of_waterfall_plots):
    explainer = shap.Explainer(bst)
    if clf.translate:
        shap_values = explainer(clf.test_data[IRL_FEATURE_NAMES])
    else:
        shap_values = explainer(clf.test_data[FEATURES_LIST])

    if clf.test_data.shape[0] < no_of_waterfall_plots:
        print_list = [i for i in range(0, clf.test_data.shape[0])]
    else:
        print_list = [i for i in range(0, no_of_waterfall_plots)]

    for i in print_list:
        shap.plots.waterfall(shap_values[i], show=False)
        plt.savefig(f'shap_waterfall{i}.png')
        plt.clf()

    shap.plots.bar(shap_values, show=False)
    plt.savefig('shap_bar.png')
    plt.clf()
    if clf.translate:
        shap.summary_plot(shap_values, clf.test_data[IRL_FEATURE_NAMES], show=False)
    else:
        shap.summary_plot(shap_values, clf.test_data[FEATURES_LIST], show=False)
    plt.savefig('shap_summary.png')
    plt.clf()

    ax = xgb.plot_importance(bst,
                             xlabel="Weights-based F score",
                             importance_type='weight',
                             title="Weights-based Feature Importance")
    ax.figure.savefig('Weights-based_Importance.png')
    ax = xgb.plot_importance(bst,
                             xlabel="Gains-based F score",
                             importance_type='gain',
                             title="Gains-based Feature Importance")
    ax.figure.savefig('Gains-based_Importance.png')
    ax = xgb.plot_importance(bst,
                             xlabel="Coverage-based F score",
                             importance_type='cover',
                             title="Coverage-based Feature Importance")
    ax.figure.savefig('Coverage-based_Importance.png')

    ax = xgb.plot_importance(bst,
                             xlabel="Total Gains score",
                             importance_type='total_gain',
                             title="Total Gains of Features")
    ax.figure.savefig('Total-gains_Importance.png')
    ax = xgb.plot_importance(bst,
                             xlabel="Total Coverages score",
                             importance_type='total_cover',
                             title="Total Coverages of Features")
    ax.figure.savefig('Total-coverage_Importance.png')


def execute_min_frames_daily(clf, pred_test_xgb, pred_test_xgb_raw_df, plot_daily_hist_for_min_daily):
    clf.test_data_nd['pdv_pred'] = pred_test_xgb
    clf.test_data_nd['mday_id'] = np.empty(clf.test_data_nd.shape[0])
    clf.test_data_nd['hms'] = np.empty(clf.test_data_nd.shape[0])
    clf.test_data_nd['dateTimeLocal'] = np.empty(clf.test_data_nd.shape[0])

    for i in clf.test_data_nd.index:
        clf.test_data_nd.at[i, 'dateTimeLocal'] = datetime.datetime.strptime(
            clf.test_data_nd['dateTime'][i], DATETIMEFORMAT1) - datetime.timedelta(hours=4)

        clf.test_data_nd.at[i, 'mday_id'] = make_daily_id(
            clf.test_data_nd['dateTimeLocal'][i], DATETIMEFORMAT1)

        clf.test_data_nd.at[i, 'hms'] = '{}:{}:{}'.format(
            clf.test_data_nd['dateTimeLocal'][i].hour, clf.test_data_nd['dateTimeLocal'][i].minute, clf.test_data_nd['dateTimeLocal'][i].second)

    unique_day = pd.unique(clf.test_data_nd['mday_id'])
    day_dict = {}
    for dayid in unique_day:

        day_dict[dayid] = count1s(
            clf.test_data_nd[clf.test_data_nd['mday_id'] == dayid]['pdv_pred'].tolist())

        if day_dict[dayid] < int(clf.min_frames_daily):
            id_to_extract = clf.test_data_nd[clf.test_data_nd['mday_id']
                                             == dayid].index
            id_to_extract = np.array(id_to_extract).tolist()
            pred_test_xgb_partial = pred_test_xgb_raw_df['pred_proba'].loc[id_to_extract].copy(
            ).tolist()

            if plot_daily_hist_for_min_daily:
                plt.hist(pred_test_xgb_partial)
                plt.title('pred_proba score distribution on {}'.format(
                    daily_id_to_datetime(dayid, DATETIMEFORMAT1)))
                plt.xlabel('pred_proba')
                plt.ylabel('count')
                plt.show()

            pred_test_xgb_partial_modified = pred_test_xgb_partial.copy()

            for ij in range(0, len(pred_test_xgb_partial)):
                if pred_test_xgb_partial[ij] > clf.limit_floor_thr:
                    pred_test_xgb_partial_modified[ij] = clf.limit_floor_thr + \
                        clf.step_size
            for ii in range(0, len(pred_test_xgb_partial_modified)):

                clf.input_data['pdv_pred'][id_to_extract[ii]
                                           ] = pred_test_xgb_partial_modified[ii]

    clf.input_data = clf.input_data.merge(pred_test_xgb_raw_df['pred_proba'],
                                          how='left',
                                          left_on='index',
                                          right_on='index_orig')
    clf.input_data['pdv_pred'] = proba_to_pred(
        clf.input_data['pdv_pred'], clf.limit_floor_thr)

    return clf.input_data, clf.test_data_nd


def std_threshold_for_flatline(data, tol_diff):
    k = tol_diff
    lenn = len(data)
    if lenn % 2 == 1:
        n = (lenn-1) / 2
        a = np.min(data)
        return np.sqrt(k**2 * (2*(n**3) + 3*(n**2) + n) / ((2*n+1)**3))
    else:
        return 0.5*tol_diff


def download_data_from_bucket(bucket_name, filepath):
    s3_resource = boto3.resource('s3')
    obj = s3_resource.Object(bucket_name, filepath)
    load_data = obj.get()['Body'].read().decode('utf-8')

    output = stringToArray(load_data)
    return output


def stringToArray(X):
    split_data = X.split('\n')
    row = len(split_data)-1
    col = len(split_data[0].split(','))
    output = np.zeros((row, col))
    for i in range(row):
        output[i, :] = split_data[i].split(',')
    return output


def moving_average(time, val, window_size):
    """
    If window_size==0: returns input
    """
    if window_size == 0:
        mva_time = time
        mva_val = val
    else:
        temp = pd.Series(val)
        temp_mva = temp.rolling(window=window_size, center=True).mean()
        mva_val = temp_mva[int(math.floor(window_size/2)):(len(temp)-int(math.floor(window_size/2)))]
        mva_val = np.array(mva_val)

        mva_time = time[int(math.floor(window_size/2)):(len(time)-int(math.floor(window_size/2)))]
    return mva_time, mva_val


def std_threshold_for_flatline(data, tol_diff):
    k = tol_diff
    lenn = len(data)
    if lenn % 2 == 1:
        n = (lenn-1) / 2
        a = np.min(data)
        return np.sqrt(k**2 * (2*(n**3) + 3*(n**2) + n) / ((2*n+1)**3))
    else:
        return 0.5*tol_diff


def check_need_merge(data):
    n = len(data)
    for i in range(0, n-1):
        if data[i] != 0 and data[i+1] != 0:
            return 1

    return 0


def flatline_checker(user_id, startDateTime):

    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    s3_resource = boto3.resource('s3')

    dbPrimaryKey = 'userId'
    dbSortKey = 'dateTime'

    ts_spo2 = 0.04
    ts_rr = 0.06

    roll_length = 4
    flatline_threshold = 0.2
    flatline_tol_diff = 1
    min_flatline_length = 1
    min_flatline_length = int(min_flatline_length / ts_rr)

    fix_000 = False

    ScanIndexForward = True

    table = dynamodb.Table('respiree-data-processing-6lmz7bq82k709j7')

    windowMvaRR = 21
    windowMvaHR = 5

    listMode = ['HR', 'RR']

    queryResponse = table.query(
        KeyConditionExpression=Key(dbPrimaryKey).eq(user_id) & Key(
            dbSortKey).between(str(startDateTime), str(startDateTime)),
        ScanIndexForward=ScanIndexForward
    )

    Items = queryResponse['Items']

    for i in range(queryResponse['Count']):

        tempFilepath = Items[i]['filepath']
        tempBucket = Items[i]['bucket']
        tempMode = Items[i]['dashboardMode']
        tempFilename = Items[i]['filename'].split('.')[0]
        tempHardwareMode = Items[i]['hardwareMode']

        dateTimeSensor = Items[i]['dateTimeSensor']

        if tempMode not in listMode:
            continue

        data = download_data_from_bucket(tempBucket, tempFilepath)

        if fix_000:
            if data[-1, 0] == 0:
                data = data[0:-1, :]

        seqNumber = data[:, 0]
        sensor1 = data[:, 1]
        sensor2 = data[:, 2]

        if tempMode == 'RR':
            windowMva = windowMvaRR
        elif tempMode == 'HR':
            windowMva = windowMvaHR

        if tempHardwareMode == 'respiratory-rate':
            ts = ts_rr
        elif tempHardwareMode == 'pulse-oximetry':
            ts = ts_spo2

        timestamp = np.arange(len(sensor1)) * ts
        mva_time_s1, mva_val_s1 = moving_average(timestamp, sensor1, windowMva)
        mva_time_s2, mva_val_s2 = moving_average(timestamp, sensor2, windowMva)

        flatline_index = 2
        flatline_window_length = math.ceil(
            flatline_threshold * (timestamp[-1]))

        th_arr = np.zeros(int(len(timestamp) - roll_length + 1))
        fl_arr = np.zeros(int(len(timestamp) - roll_length + 1))

        for k in range(0, int(len(timestamp) - roll_length + 1)):
            this_window = sensor1[k:k + roll_length]
            if np.std(this_window) <= std_threshold_for_flatline(this_window, flatline_tol_diff):
                fl_arr[k] = 0
            else:
                fl_arr[k] = 1

        fl_lengths = np.full(len(fl_arr), 0)
        fl_lengths = fl_lengths.tolist()
        fl_arr = fl_arr.tolist()

        for k in range(0, len(fl_lengths)):
            if fl_arr[k] == 0:
                fl_lengths[k] = 1

        n_fl_arr_init = len(fl_arr)
        mergecount = 0

        while(check_need_merge(fl_lengths)):
            k = 0
            while k < n_fl_arr_init - mergecount - 1:

                if fl_lengths[k] != 0:
                    if fl_arr[k] == 0 and fl_arr[k+1] == 0:
                        fl_lengths[k] += 1
                        fl_lengths.pop(k+1)
                        fl_arr.pop(k+1)
                        mergecount += 1
                        k -= 1

                k += 1

        sum_flatline = 0
        nonzero_counter = 0
        for k in range(0, len(fl_lengths)):
            if fl_lengths[k] >= min_flatline_length:
                sum_flatline += fl_lengths[k]
                nonzero_counter += 1

        fl_positions = np.zeros(len(fl_lengths))
        fl_lengths_1_filled = fl_lengths.copy()
        for j in range(0, len(fl_lengths)):
            if fl_lengths_1_filled[j] == 0:
                fl_lengths_1_filled[j] = 1

        for i in range(0, len(fl_lengths)):
            for j in range(0, i):
                fl_positions[i] += fl_lengths_1_filled[j]

        return (sum_flatline + 2*nonzero_counter) / (n_fl_arr_init + 2)
