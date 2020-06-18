from tqdm import tqdm
import gc
import pandas as pd
import sys
import numpy as np
sys.path.append('../../')


tqdm.pandas()


def calc_days_from_event(days, current_day):
    """
    前にeventがあった日からの経過日数
    Args:
        days (list): イベントがあった日付のリスト
        current_day (int): 現在の日付
    Returns:
        [type]: [description]
    """

    day_diff_list = np.asarray(days) - current_day

    if min(day_diff_list) > 0:  # current_dayより前にeventがない場合
        return np.nan
    else:
        day_diff = day_diff_list[day_diff_list <= 0].max()  # 0より小さい=current_dayより前
        return abs(day_diff)  # 絶対値で返す


def calc_days_until_event(days, current_day):
    """
    次にeventがある日までの日数
    Args:
        days (list): イベントがあった日付のリスト
        current_day (int): 現在の日付
    Returns:
        [type]: [description]
    """

    day_diff_list = np.asarray(days) - current_day

    if max(day_diff_list) < 0:  # current_dayより後にeventがない場合
        return np.nan
    else:
        day_diff = day_diff_list[day_diff_list >= 0].min()  # 0より小さい=current_dayより前
        return abs(day_diff)  # 絶対値で返す


print('read_transformed...')
df_all = pd.read_pickle('../../scaled_35093990_33386550_melt_over0sellprice.pkl')
print('tday')
date_tday = dict(zip(df_all.date.unique(), np.arange(len(df_all.date.unique()))))
df_all['tday'] = df_all["date"].apply(lambda x: date_tday[x])
print(df_all.shape)


from_cols = []
until_cols = []
df_snap = df_all[['tday', 'date']].drop_duplicates()
for col in ['snap_CA', 'snap_TX', 'snap_WI']:
    df_tday = df_all[['tday', col, 'date']]
    df_tday = df_tday.drop_duplicates()
    print(df_tday.shape)

    snap_days = df_tday.tday[df_tday[col] == 1].tolist().copy()
    from_col = f'day_from_{col}'
    until_col = f'day_until_{col}'
    df_tday[from_col] = df_tday["tday"].progress_apply(lambda x: calc_days_from_event(snap_days, x))
    df_tday[until_col] = df_tday["tday"].progress_apply(lambda x: calc_days_until_event(snap_days, x))

    df_tday['date_new'] = pd.to_datetime(df_tday['date'])

    if col in ['snap_CA', 'snap_TX']:
        df_tday[until_col][df_tday[until_col].isnull()] = df_tday["date_new"].apply(lambda x: 31 - x.day)[df_tday[until_col].isnull()]
    else:
        df_tday[until_col][df_tday[until_col].isnull()] = df_tday["date_new"].apply(lambda x: 32 - x.day)[df_tday[until_col].isnull()]
    df_snap = pd.merge(df_snap, df_tday, on=['tday', 'date'])
    from_cols.append(from_col)
    until_cols.append(until_col)
use_col = ['date'] + from_cols + until_cols
df_snap = df_snap[use_col]
print(df_snap.tail(10))
print(df_snap.head())

df_event = df_all[['tday', 'date', 'event_name_1']].drop_duplicates()
event_days = df_event.tday[df_event.event_name_1.notnull()].tolist().copy()
df_event["day_from_event"] = df_event["tday"].apply(lambda x: calc_days_from_event(event_days, x))
df_event["day_until_event"] = df_event["tday"].apply(lambda x: calc_days_until_event(event_days, x))
df_event = df_event[['date', 'day_from_event', 'day_until_event']]
print(df_event.tail(10))
print(df_event.head())

df_fe = df_all[['id', 'date']]
df_fe = pd.merge(df_fe, df_snap, on=['date'], how='left')
df_fe = pd.merge(df_fe, df_event, on=['date'], how='left')

print(df_fe.tail(10))
print(df_fe.head())
print(df_fe.shape)

df_fe.to_pickle('f_be_af.pkl')
