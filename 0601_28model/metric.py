# https://github.com/shaoroon/kaggle_m5/blob/master/Utils/metric.py

# ==================================================================================================================
# sakamiさん実装の改良
from typing import Union

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm_notebook as tqdm


class WRMSSEEvaluator(object):
    # インスタンスの作成
    # val_preds以外のデータを与え、計算の準備をする
    # train_df：sales_train_validationの、train部分全体（ID列含む）
    # valid_df：sales_train_validationの、val部分全体(ID列は含まない)
    # calendar：元データのcalendar.csvそのまま（と思う）
    # prices：元データのsell_prices.csvそのまま（と思う）
    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')] # dから始まる列のみを抽出（ID列を落とす）
        train_target_columns = train_y.columns.tolist() # trainのカラム名をリストで取得
        weight_columns = train_y.iloc[:, -28:].columns.tolist() # weightに関係のある最後28の列名だけ取得

        train_df['all_id'] = 0  # 後で全体を1グループとしてAggするための列を作成

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist() #id列を抽出（~は以外という意味）
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist() #validの列名を追加

        if not all([c in valid_df.columns for c in id_columns]): #id_columnsの中に、valid_dfに存在する列があるかどうかをチェック（列の重複チェック）
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False) #valid_dfに、id列を追加する
        # else:
        #     #例外処理（自分で追加）
        #     print("列が重複しています（valid & id）")
        #     exit

        #インスタンス変数の作成
        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices
        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        #重みのdfを作成
        weight_df = self.get_weight_df() #get_weight_dfはクラスで定義している関数。

        # Aggするためのグループを定義
        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            # 属性の設定
            # fは.formatと同じ。lv1_train_dfみたいな形。group_idの組み合わせをlevelとしている。
            #group_idをキーにgroupbyして、sell数をsumしている
            # group_id＆日ごとの、売り上げ数が算出される。trainはscaleの計算に、validはRMSEの計算に使われる

            # setattr(self, f'lv{i + 1}_train_df', train_df.groupby(group_id)[train_target_columns].sum()) 
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

            #（追記）scaleの調整（後で計算に使用するので列を追加しておく）
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            train_y["not_count_days"] = train_y.apply(lambda x:np.argmax(x.values != 0),axis=1) #販売が0だった日数
            train_y["count_days"] = len(train_y.columns) - train_y["not_count_days"] # 販売してからの日数する日付
            train_y["scale"] = train_y.apply(lambda x:((x.iloc[x["not_count_days"]+1:-2].values - x.iloc[x["not_count_days"]:-3].values)**2).sum()\
            / (x["count_days"]-1) ,axis=1).values #scaleの計算。販売後期間のみで差分計算。count_days-1で割る(元がn-1で計算されるため)
            setattr(self, f'lv{i + 1}_train_df', train_y) 

            #各levelごとにweightを出す。販売金額（個数×単価）をgroupごとに合計し、それが大きいほどweightが高い。
            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1) #group化したときのレコード数だけweightが算出される。（state_idなら3個）
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum()) #levelごとにweightを割合化して、属性に追加
        # initを実行し終えると、level内のgroupごとのweightが算出された状態になる

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict() #{'d_1': 11101}のように、dとweek_idの対応表を作る
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id']) #重み計算に使う列の値を取得してdfに
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'}) #stackでd_x列を分解して行にする（meltと同じ効果）
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week) #辞書に従って、wm_yr_wk列を追加

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk']) #priceを追加
        weight_df['value'] = weight_df['value'] * weight_df['sell_price'] #個数が入っていたvalue列に、sell_priceを掛ける（価格で重みづけ）
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value'] #dを列、item_idとstore_idを行にして集計
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True) #item_idとstore_id列を落とす
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False) #id列を足す
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series: #行がgroup_id、列が日付のテーブルとlevelが引数。
        train_y = getattr(self, f'lv{lv}_train_df')
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1) #誤差2乗の平均を計算。groupの数だけ計算される。

        #scaleの計算
        # scale = ((train_y.iloc[:, 1:].values - train_y.iloc[:, :-1].values) ** 2).mean(axis=1) #scaleの計算。(Yt - Yt-1)^2のmeanを取得。meanはgroupの数だけできる。
        scale = train_y["scale"].values #修正後

        return (score / scale).map(np.sqrt) #RMSSEの計算（score=の部分で誤差2乗の平均を計算しているので、1/hは不要）groupの数だけRMSSEが算出される。

    #スコアの算出
    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape #valid_dfとvalid_predsのshapeがあっているかを確認

        if isinstance(valid_preds, np.ndarray): #valid_predsの型がndarrayならdf化
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False) #id列を付ける

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1) #rmsseの算出。
            weight = getattr(self, f'lv{i + 1}_weight') #Levelごとのweightをロードする
            # lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1) #weightとlv_scoreをconcatして積を出している（prodは積を出す関数）
            lv_scores = pd.concat([weight.sort_index(), lv_scores.sort_index()], axis=1, sort=False).prod(axis=1) #★念のためindexでどちらもsort
            # lv_scores = pd.merge(lv_weight.reset_index(),lv_scores.reset_index(),how="left",on="state_id").set_index("state_id").prod(axis=1)
            all_scores.append(lv_scores.sum()) #算出したスコアをall_scoresに入れる

        return np.mean(all_scores) #AggLevel間でweightは等しいので、meanを計算

    def feval(self, preds, dtrain):
        preds = preds.reshape(self.valid_df[self.valid_target_columns].shape)
        score = self.score(preds)
        return 'WRMSSE', score, False


# train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
# train_fold_df = train_df.iloc[:, :-28]
# valid_fold_df = train_df.iloc[:, -28:]
# valid_preds = valid_fold_df.copy() + np.random.randint(100, size=valid_fold_df.shape)

# evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)
# evaluator.score(valid_preds)

# #getattrで、要素を取り出すことも可能
# getattr(evaluator,"lv10_weight")


# ==================================================================================================================