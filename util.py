import main_plot_ver2
import pandas as pd
import numpy as np
import os
import random

filename = 'C:\\Users\\Timothy Antoni\\Documents\\programming\\sampled_all_data_to_check.csv'


df = pd.read_csv(filename)[['userId', 'dateTime']]

df['userId'] = df['userId'].astype(str)
df['dateTime'] = df['dateTime'].astype(str)

# mode 1 -- plot all rows in the file
for i in range(0, df.shape[0]):
    # print(i)
    main_plot_ver2.plotting(df.iloc[i, 0], df.iloc[i, 1])
    print(i+1, 'out of', df.shape[0])


# mode 2 -- plot specific rows in excel
# k=no of row in excel.

# k = np.arange(2, 765)
# counter = 0
# for i in k:
#     main_plot_ver2.plotting(df.iloc[i-2, 0], df.iloc[i-2, 1])
#     counter += 1
#     print(counter, 'out of', len(k))


# mode 3 -- plot only the rows with remark

# dfc = df.dropna(subset=['remark'])
# dfc[['userId', 'dateTime']].to_csv('feature_lookup_list.csv', index=False)
# for i in range(0, dfc.shape[0]):
#     main_plot_ver2.plotting(dfc.iloc[i, 0], dfc.iloc[i, 1])
#     print(i+1, 'out of', dfc.shape[0])


# mode 4 -- single random frame, not from file

# main_plot_ver2.plotting('157', '2022-08-27 21:13:27')
