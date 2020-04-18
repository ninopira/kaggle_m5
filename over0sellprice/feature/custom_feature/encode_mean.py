"""
https://www.kaggle.com/kyakovlev/m5-custom-features/output?scriptVersionId=32152345
注意: holdoutの前提。cvの際にはcvごとに算出する必要あり
"""

import numpy as np
import pandas as pd
import os, sys, gc, warnings, psutil, random
import pandas as pd
import sys
sys.path.append('../../')

warnings.filterwarnings('ignore')

print('read_transformed...')
grid_df = pd.read_pickle('../../23965140_22257700_melt_over0sellprice.pkl')
print(grid_df.shape)


TARGET = 'demand'  # Our Target
END_TRAIN = 1913  # And we will use last 28 days as validation
END_TARIN = "2016-04-24"
remove_features = ['id', 'date', TARGET, 'part']

# Note: holdoutなので、3-27まででencoding
grid_df[TARGET][grid_df['date'] > ('2016-03-27')] = np.nan
base_cols = list(grid_df)

icols = [
            ['state_id'],
            ['store_id'],
            ['cat_id'],
            ['dept_id'],
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            ['item_id'],
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
            ]

for col in icols:
    print('Encoding', col)
    col_name = '_'+'_'.join(col)+'_'
    grid_df['enc'+col_name+'mean'] = grid_df.groupby(col)[TARGET].transform('mean').astype(np.float16)
    grid_df['enc'+col_name+'std'] = grid_df.groupby(col)[TARGET].transform('std').astype(np.float16)

keep_cols = [col for col in list(grid_df) if col not in base_cols]
grid_df = grid_df[['id', 'date']+keep_cols]

print(grid_df.head())
print(grid_df.tail())
print(grid_df.shape)

grid_df.to_pickle('f_hold_out_mean_encoding_df.pkl')
