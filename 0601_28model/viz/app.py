"""
train期間とsub期間の系列の確認
"""

import streamlit as st
import datetime
import time

from matplotlib import pyplot as plt
import pandas as pd


def main():
    st.title("予測値の確認")

    st.header('read_data')
    # Note: データん準備で必要。初回はこのボタンを押さないとerrorが出る
    if st.button('READ DATA AND MAKE PKL(初回のみ実行)'):
        t0 = time.time()
        calendar = pd.read_csv('../../new_input/calendar.csv')
        sales_train_validation = pd.read_csv('../../new_input/sales_train_evaluation.csv')[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
        sell_prices = pd.read_csv('../../new_input/sell_prices.csv')
        t1 = time.time()
        st.text('read_other_data:{0}'.format(t1-t0) + '[sec]')
        # st.dataframe(calendar)
        # st.dataframe(sales_train_validation)
        # st.dataframe(sell_prices)

        st.text('read_and_mod_sub_x1.0...')
        df_sub_final = pd.read_csv('./before_1.4x_test.csv')
        df_sub_final.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959', 'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']
        df_sub_final = pd.melt(df_sub_final, id_vars=['id'], var_name='day', value_name='demand')
        df_sub_final = pd.merge(df_sub_final, calendar[['d', 'date']], how='left', left_on=['day'], right_on=['d'])
        t1 = time.time()
        st.text('read_sub_x1.0_data:{0}'.format(t1-t0) + '[sec]')
        st.text(df_sub_final.shape)

        st.text('read_and_mod_sub_x1.4...')
        df_sub_final_14 = pd.read_csv('./sub_multiple_14.csv')
        df_sub_final_14.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959', 'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']
        df_sub_final_14 = pd.melt(df_sub_final_14, id_vars=['id'], var_name='day', value_name='demand')
        df_sub_final_14 = pd.merge(df_sub_final_14, calendar[['d', 'date']], how='left', left_on=['day'], right_on=['d'])
        t1 = time.time()
        st.text('read_sub_x1.4_data:{0}'.format(t1-t0) + '[sec]')
        st.text(df_sub_final_14.shape)
        # st.dataframe(df_sub_final)
        df_sub_final['demand_1.4'] = df_sub_final_14['demand']
        df_sub_final = df_sub_final.merge(sales_train_validation, how='left', on=['id'])
        st.text('write_pickle...')
        df_sub_final.to_pickle('df_sub_final.pkl')

        st.text('read_train_data_pkl...')
        df_all = pd.read_pickle('../35093990_33386550_melt_over0sellprice.pkl')[['id', 'date', 'demand']]
        df_all['date'] = pd.to_datetime(df_all['date'])
        df_all = df_all.query('date > "2016-02-28"')
        df_all = df_all.query('date <= "2016-05-22"')
        sales_train_validation = pd.read_csv('../../new_input/sales_train_evaluation.csv')[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
        df_all = df_all.merge(sales_train_validation, how='left', on=['id'])
        st.text('write_pickle...')
        df_all.to_pickle('df_all_small.pkl')


    st.text('read_pkl...')
    df_sub_final = pd.read_pickle('df_sub_final.pkl')

    st.text('sub_tail')
    st.dataframe(df_sub_final.tail(100))

    st.text('read_train_data_pkl...')
    df_all = pd.read_pickle('./df_all_small.pkl')
    st.text('all_data_tail')
    st.dataframe(df_all.tail(100))

    selected_depts = st.multiselect('select dept', df_sub_final['dept_id'].unique())
    selected_items = st.multiselect('select item', df_sub_final[df_sub_final['dept_id'].isin(selected_depts)]['item_id'].unique())

    st.header('visualize')
    viz(df_sub_final, df_all, selected_items)


def viz(df_sub_final, df_all, selected_items):
    fig = viz_pred(df_sub_final, ['demand', 'demand_1.4'], title='all')
    st.pyplot(fig)

    fig = filter_item_id_viz_pred(df_sub_final, ['demand', 'demand_1.4'], selected_items)
    st.pyplot(fig)

    figs = filter_item_id_viz_train_pred(df_sub_final, df_all, ['demand', 'demand_1.4'], selected_items)
    for fig in figs:
        st.pyplot(fig)


def viz_pred(df, cols, title=None):
    fig = plt.figure(figsize=(40, 16))
    ax = fig.add_subplot(1, 1, 1)
    df['date'] = pd.to_datetime(df['date'])
    df_tmp = df.groupby('date')[cols].sum().reset_index()
    for target in cols:
        ax.plot(df_tmp['date'], df_tmp[target], label=target, marker="o")
        ax.legend(fontsize='20')
    ax.axvline(datetime.datetime(2016, 5, 30), color='blue', linestyle='--', alpha=0.4) # Memorial day
    ax.axvline(datetime.datetime(2016, 6, 2), color='green', linestyle='--', alpha=0.4) # NBA
    ax.axvline(datetime.datetime(2016, 6, 7), color='orange', linestyle='--', alpha=0.4) # ラマダン
    ax.axvline(datetime.datetime(2016, 6, 19), color='green', linestyle='--', alpha=0.4) # NBA

    if title is not None:
        ax.set_title(title, fontsize='50')
    return fig


def filter_item_id_viz_pred(df, cols, item_ids):
    fig = plt.figure(figsize=(40, 16))
    ax = fig.add_subplot(1, 1, 1)
    for elm in item_ids:
        df_tmp = df[df['item_id'] == elm]
        df_tmp['date'] = pd.to_datetime(df_tmp['date'])
        df_tmp = df_tmp.groupby('date')[cols].sum().reset_index()
        for target in cols:
            ax.plot(df_tmp['date'], df_tmp[target], label=f'{elm}_{target}', marker="o")
            ax.legend(fontsize='20')
    ax.axvline(datetime.datetime(2016, 5, 30), color='blue', linestyle='--', alpha=0.4) # Memorial day
    ax.axvline(datetime.datetime(2016, 6, 2), color='green', linestyle='--', alpha=0.4) # NBA
    ax.axvline(datetime.datetime(2016, 6, 7), color='orange', linestyle='--', alpha=0.4) # ラマダン
    ax.axvline(datetime.datetime(2016, 6, 19), color='green', linestyle='--', alpha=0.4) # NBA
    ax.set_title(item_ids, fontsize='50')
    return fig


def filter_item_id_viz_train_pred(df_pred, df_all, cols, item_ids):
    figs = []
    for elm in item_ids:
        fig = plt.figure(figsize=(40, 16))
        ax = fig.add_subplot(1, 1, 1)
        df_pred_tmp = df_pred[df_pred['item_id'] == elm]
        df_pred_tmp['date'] = pd.to_datetime(df_pred_tmp['date'])
        df_pred_tmp = df_pred_tmp.groupby('date')[cols].sum().reset_index()
        for target in cols:
            ax.plot(df_pred_tmp['date'], df_pred_tmp[target], label=f'{elm}_{target}', marker="o")
            ax.legend(fontsize='20')
        df_all_tmp = df_all[df_all['item_id'] == elm]
        df_all_tmp['date'] = pd.to_datetime(df_all_tmp['date'])
        df_all_tmp = df_all_tmp.groupby('date')['demand'].sum().reset_index()
        ax.plot(df_all_tmp['date'], df_all_tmp['demand'], label=f'{elm}_TRUE', marker="o")
        ax.legend(fontsize='20')
        ax.set_title(elm, fontsize='50')
        figs.append(fig)
    return figs


if __name__ == "__main__":
    main()
