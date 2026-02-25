import pandas as pd
from scipy import stats

class StatisticsEngine:
    @staticmethod
    def get_group_stats(df: pd.DataFrame, group_col: str, target_col: str):
        return df.groupby(group_col)[target_col].agg([
            'count', 'mean', 'std', 'min', 
            lambda x: x.quantile(0.25), 'median', lambda x: x.quantile(0.75), 'max'
        ]).rename(columns={'<lambda_0>': 'Q1', '<lambda_1>': 'Q3'}).round(2)

    @staticmethod
    def perform_t_test(df: pd.DataFrame):
        smokers = df[df['smoker'] == 'yes']['charges']
        non_smokers = df[df['smoker'] == 'no']['charges']
        t_stat, p_val = stats.ttest_ind(smokers, non_smokers, equal_var=False)
        return t_stat, p_val