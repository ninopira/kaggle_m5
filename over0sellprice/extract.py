import os
import time

import pandas as pd

from reduce_mem import reduce_mem_usage


########################
print('########################')
# read_transfomed
print('read_transfomed_data')
t0 = time.time()
df_all = pd.read_pickle('../20200327_baseline/23965140_22257700_melt.pkl')
df_all = reduce_mem_usage(df_all)
print(df_all.shape)
df_all = df_all.query('sell_price > 0')
print(df_all.shape)
t1 = time.time()
print('read_transfomed_data:{0}'.format(t1-t0) + '[sec]')
print('########################')
########################

print('writing...')
df_all.to_pickle('23965140_22257700_melt_over0sellprice.pkl')