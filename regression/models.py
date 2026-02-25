import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

class InsuranceModeler:
    """Բազմակի ռեգրեսիայի և վիճակագրական հաշվարկների մոդուլ"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self._prepare_data()

    def _prepare_data(self):
        """Տվյալների նախապատրաստում և կոդավորում"""

        le = LabelEncoder()
        for col in ['sex', 'smoker', 'region']:
            if col in self.df.columns:
                self.df[f'{col}_enc'] = le.fit_transform(self.df[col])
        
    def get_group_statistics(self):
        """Հիմնական վիճակագրություն ըստ ծխողների"""

        return self.df.groupby('smoker')['charges'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)

    def run_regression(self):
        """Բազմակի գծային ռեգրեսիա (OLS)"""

        features = ['age', 'bmi', 'children', 'smoker_enc', 'sex_enc']
        X = self.df[features]
        X = sm.add_constant(X)
        y = self.df['charges']
        
        self.model = sm.OLS(y, X).fit()
        return self.model

    def get_correlation_matrix(self):
        """Կորելյացիոն մատրիցա միայն թվային սյուների համար"""
        
        return self.df.select_dtypes(include=[np.number]).corr()