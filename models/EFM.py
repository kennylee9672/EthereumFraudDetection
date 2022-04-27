import pandas as pd
from sklearn.utils import shuffle


class Ethereum_Fraud_Model:
    """
        Data: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset
    """

    def __init__(self):
        self.import_data()
        self.preprocess_data()
        self.models = {}

    def import_data(self, name='Ethereum'):
        if name == 'Ethereum':
            self.df = pd.read_csv('./data/transaction_dataset.csv', index_col=0)
        elif name == 'Ethereum_Filled':
            self.df = pd.read_csv('./data/transaction_dataset_filled.csv', index_col=0)
        self.df = shuffle(self.df)

    def preprocess_data(self):
        self.filtered_colns = ['Index', 'Address', ' ERC20 most sent token type', ' ERC20_most_rec_token_type']

    def add_model(self, model):
        self.models[model.get_model_name().lower()] = model

    def run_model(self, model_name):
        if model_name.lower() in self.models.keys():
            self.models[model_name.lower()].run()
        else:
            print('No Associated Model Found!')

    def run_models(self):
        for key in self.models:
            self.models[key].run()