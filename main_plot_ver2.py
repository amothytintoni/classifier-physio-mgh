
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import boto3
from boto3.dynamodb.conditions import Key, Attr

OUTPUT_TO_CSV = True


def plotting(id, jikan):
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    s3_resource = boto3.resource('s3')

    dbPrimaryKey = 'userId'
    dbSortKey = 'dateTime'

    plot_graph = True

    fix_out_of_range_s1 = False
    fix_out_of_range_s2 = False

    fix_000 = False

    ScanIndexForward = True

    table = dynamodb.Table('respiree-data-processing-6lmz7bq82k709j7')
    user_id = id
    startDateTime = jikan

    windowMvaRR = 21
    windowMvaHR = 5

    ts_spo2 = 0.04
    ts_rr = 0.06

    listMode = ['HR', 'RR']

    queryResponse = table.query(
        KeyConditionExpression=Key(dbPrimaryKey).eq(user_id) & Key(
            dbSortKey).between(str(startDateTime), str(startDateTime)),
        ScanIndexForward=ScanIndexForward
    )

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
        import pandas as pd
        import math
        """
    If window_size==0: returns input
    """
        if window_size == 0:
            mva_time = time
            mva_val = val
        else:
            temp = pd.Series(val)
            temp_mva = temp.rolling(window=window_size, center=True).mean()
            mva_val = temp_mva[int(math.floor(window_size/2))
                                   :(len(temp)-int(math.floor(window_size/2)))]
            mva_val = np.array(mva_val)

            mva_time = time[int(math.floor(window_size/2))
                                :(len(time)-int(math.floor(window_size/2)))]
        return mva_time, mva_val

    Items = queryResponse['Items']
    counter = 0

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

        acclX = data[:, 3]
        acclY = data[:, 4]
        acclZ = data[:, 5]

        temperature = data[:, 6]

        if fix_out_of_range_s1:
            sensor1[sensor1 > 2000] = np.median(sensor1)
            sensor1[sensor1 < 15] = np.median(sensor1)

        if fix_out_of_range_s2:
            sensor2[sensor2 > 2000] = np.median(sensor2)
            sensor2[sensor2 < 15] = np.median(sensor2)

        diffSeq = np.zeros(len(seqNumber))
        diffSeq = seqNumber[1:len(seqNumber)]-seqNumber[0:-1]

        if tempMode == 'RR':
            windowMva = windowMvaRR
        elif tempMode == 'HR':
            windowMva = windowMvaHR

        if tempHardwareMode == 'respiratory-rate':
            ts = ts_rr
        elif tempHardwareMode == 'pulse-oximetry':
            ts = ts_spo2

        timestamp = np.arange(len(sensor1))*ts
        mva_time_s1, mva_val_s1 = moving_average(timestamp, sensor1, windowMva)
        mva_time_s2, mva_val_s2 = moving_average(timestamp, sensor2, windowMva)

        if plot_graph:

            if tempMode == 'RR':
                plt.figure(tempFilename)

                plt.subplot(211)
                plt.plot(timestamp, sensor1, label='sensor1')
                plt.plot(mva_time_s1, mva_val_s1)
                plt.legend()
                plt.title('{}\nrecords = {}'.format(dateTimeSensor, len(data)))

                plt.subplot(212)
                plt.plot(timestamp, sensor2, label='ACCL all')
                plt.plot(mva_time_s2, mva_val_s2)
                plt.legend()

            elif tempMode == 'HR':
                plt.figure(tempFilename)

                plt.subplot(211)
                plt.plot(timestamp, sensor1, label='sensor1')
                plt.plot(mva_time_s1, mva_val_s1)
                plt.legend()
                plt.title('{}\nrecords = {}'.format(dateTimeSensor, len(data)))

                plt.subplot(212)
                plt.plot(timestamp, sensor2, label='sensor2')
                plt.plot(mva_time_s2, mva_val_s2)
                plt.legend()

        counter += 1
        print('Process {}/{}: {}: length {}'.format(counter,
              queryResponse['Count'], tempFilename, len(sensor1)))

        plt.show()

        if OUTPUT_TO_CSV:
            df_outp = pd.DataFrame(data=[timestamp, sensor1, mva_time_s1, mva_val_s1]).transpose().to_csv(
                'rawss1_valid.csv', index=False)
            plt.clf()
