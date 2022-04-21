from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

from config import TARGET


class PCA_Transformer:
    def __init__(self, df):
        self.df = df
        self.features = df.columns
        self.n_components = min(4, len(self.features) - 1)

    def set_n_components(self, n):
        self.n_components = n

    def update_dataframe(self, df):
        self.df = df
        self.features = df.columns
        self.n_components = min(4, len(self.features) - 1)

    def run(self):
        # Separate features and target
        X = self.df.loc[:, self.features].values
        y = self.df[TARGET]

        # Standardize the features
        X = StandardScaler().fit_transform(X)

        # Construct PCA
        pca = PCA(n_components=self.n_components)
        components = pca.fit_transform(X)

        # Concat new features with target
        columns = ['pc_' + str(i + 1) for i in range(self.n_components)]
        df_pca = pd.DataFrame(data=components, columns=columns)
        df_pca[TARGET] = y

        return df_pca

    def fillna(self):
        self.df.fillna(self.df.median(), inplace=True)