import pandas as pd
import sys
import scipy
import numpy as np
sys.path.append('../../')

print('read_transformed...')
df_all = pd.read_pickle('../../23965140_22257700_melt_over0sellprice.pkl')
print(df_all.shape)
df_feat = df_all[['id', 'demand', 'date']]
df_feat['demand_lag_28'] = df_feat.groupby(["id"])["demand"].transform(lambda x: x.shift(28))
df_feat['demand_lag_28'] = df_feat['demand_lag_28'].fillna(0)


for val in [28]:
    colname = f"demand_lag_{val}_fft"
    df_feat[colname] = df_feat.groupby(["id"])["demand_lag_28"].transform(lambda x: np.fft.fft(x).real)
    print(colname, df_feat[colname].mean())

    colname = f"demand_lag_{val}_ifft"
    df_feat[colname] = df_feat.groupby(["id"])["demand_lag_28"].transform(lambda x: np.fft.ifft(x))
    print(colname, df_feat[colname].mean())

    colname = f"demand_lag_{val}_dct"
    df_feat[colname] = df_feat.groupby(["id"])["demand_lag_28"].transform(lambda x: scipy.fft.dct(x.values, norm='ortho'))
    print(colname, df_feat[colname].mean())

    colname = f"demand_lag_{val}_idct"
    df_feat[colname] = df_feat.groupby(["id"])["demand_lag_28"].transform(lambda x: scipy.fft.idct(x.values, norm='ortho'))
    print(colname, df_feat[colname].mean())

    colname = f"demand_lag_{val}_dst"
    df_feat[colname] = df_feat.groupby(["id"])["demand_lag_28"].transform(lambda x: scipy.fft.dst(x.values, norm='ortho'))
    print(colname, df_feat[colname].mean())

    colname = f"demand_lag_{val}_idst"
    df_feat[colname] = df_feat.groupby(["id"])["demand_lag_28"].transform(lambda x: scipy.fft.idst(x.values, norm='ortho'))
    print(colname, df_feat[colname].mean())

df_feat.drop(['demand', 'demand_lag_28'], inplace=True, axis=1)

print(df_feat.head())
print(df_feat.shape)
df_feat.to_pickle('f_id_fft_lag_demand.pkl')
