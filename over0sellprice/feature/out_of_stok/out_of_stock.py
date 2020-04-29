# https://www.kaggle.com/sibmike/m5-out-of-stock-feature-640x-faster
import pandas as pd
import sys
from tqdm import tqdm
import numpy as np

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt_over0sellprice.pkl')
print(df_all.shape)


def gap_finder(ts):
    # this function finds gaps and calculates their length:
    # note ts: 0 = day with sales, 1 = days with 0 sales
    for i, gap in enumerate(ts):
        if gap == 0:
            continue
        elif i != 0:
            ts[i] += ts[i-1]
            if ts[i-1] != 0:
                ts[i-1] = -1
    return ts


df_all['gaps'] = (~(df_all['demand'] > 0)).astype(int)
prods = list(df_all.id.unique())
s_list = []  # list to hold gaps in days
e_list = []  # list to hold expected values of gaps
p_list = []  # list to hold avg probability of no sales

total_days = 1941

for prod_id, df in tqdm(df_all.groupby("id")):
    # extract gap_series for a prod_id
    sales_gaps = df.loc[:, 'gaps']

    # calculate initial probability
    zero_days = sum(sales_gaps)
    p = zero_days/total_days

    # find and mark gaps
    accum_add_prod = np.frompyfunc(lambda x, y: int((x+y)*y), 2, 1)
    sales_gaps[:] = accum_add_prod.accumulate(df["gaps"], dtype=np.object).astype(int)
    sales_gaps[sales_gaps < sales_gaps.shift(-1)] = np.NaN
    sales_gaps = sales_gaps.fillna(method="bfill").fillna(method='ffill')
    s_list += [sales_gaps]

    # calculate E/total_days for all possible gap lengths:
    gap_length = sales_gaps.unique()

    d = {length: ((1-p**length)/(p**length*(1-p))) /
         365 for length in gap_length}
    sales_E_years = sales_gaps.map(d)

    # cut out supply_gap days and run recursively
    p1 = 0
    while p1 < p:
        if p1 != 0:
            p = p1
        # once in 100 years event; change to your taste here
        gap_days = sum(sales_E_years > 100)
        p1 = (zero_days-gap_days+0.0001)/(total_days-gap_days)
        d = {length: ((1-p1**length)/(p1**length*(1-p1))) / 365 for length in gap_length}
        sales_E_years = sales_gaps.map(d)

    # add results to list it turns out masked replacemnt is a very expensive operation in pandas, so better do it in one go
    e_list += [sales_E_years]
    p_list += [pd.Series(p, index=sales_gaps.index)]

# add it to grid_df in one go fast!:
df_all['gap_days'] = pd.concat(s_list)
df_all['gap_e'] = pd.concat(e_list)
df_all['sale_prob'] = pd.concat(p_list)

df_feat = df_all[['id', 'date', 'gap_days', 'gap_e', 'sale_prob']]

print(df_feat.head())
print(df_feat.tail())
print(df_feat.shape)
df_feat.to_pickle('f_out_of_stack.pkl')
