import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, plot_confusion_matrix, precision_recall_curve, f1_score
from sklearn.metrics import precision_score, recall_score, auc, accuracy_score
from sklearn.mixture import GaussianMixture

from config import *

from sklearn.mixture import GaussianMixture


class GMM_Model():
    """
        Class for GMM Model
    """

    def __init__(self, df):
        self.df = df
        self.preprocess_data()
        self.train, self.valid, self.test = self.spilt_data_3way()
        self.model_name = 'GMM'

    def get_model_name(self):
        return self.model_name

    def update_dataframe(self, df):
        self.df = df
        self.preprocess_data()
        self.train, self.valid, self.test = self.spilt_data_3way()

    def preprocess_data(self):
        # QuickFix, Need to update it once data cleaning and preparation step completes
        self.df.fillna(self.df.median(), inplace=True)
        self.normal_data = self.df[self.df[TARGET] == 0]
        self.anomaly_data = self.df[self.df[TARGET] == 1]

    # TO-DO: Allow consumer function to split percentages
    def spilt_data_3way(self, normal_data=None, anomaly_data=None):
        if not normal_data:
            normal_data = self.normal_data
        if not anomaly_data:
            anomaly_data = self.anomaly_data

        # Spliting Normal data into 80% + 10% + 10%
        train, valid, test = np.split(normal_data.sample(frac=1, random_state=42),
                                      [int(.8 * len(normal_data)), int(.9 * len(normal_data))])

        # Splitting Anomaly data into 50% + 50%
        valid_anomaly, test_anomaly = np.split(anomaly_data.sample(frac=1, random_state=42),
                                               [int(.5 * len(anomaly_data))])

        # Combining Validation data and shuffling it
        valid = pd.concat([valid, valid_anomaly]).sample(frac=1)

        # Combining Test Data and shuffling it
        test = pd.concat([test, test_anomaly]).sample(frac=1)
        # print('Train shape: ', train.shape)
        # print('Proportion os anomaly in training set: %.3f\n' % train[TARGET].mean())
        # print('Valid shape: ', valid.shape)
        # print('Proportion os anomaly in validation set: %.3f\n' % valid[TARGET].mean())
        # print('Test shape:, ', test.shape)
        # print('Proportion os anomaly in test set: %.3f\n' % test[TARGET].mean())
        return (train, valid, test)

    # TO-DO: Break this big function into train, test and evaluate functions
    def run(self, train=None, valid=None, test=None, verbose=False):
        print("Model Name: " + self.get_model_name())
        display(self.df.head())
        if not train:
            train = self.train
        if not valid:
            valid = self.valid
        if not test:
            test = self.test
        for components in range(1, 5):
            print('Components: ' + str(components))
            # gmm = GaussianMixture(n_components=components, n_init=4, random_state=42, covariance_type='tied')
            gmm = GaussianMixture(n_components=components, n_init=4, random_state=42)
            # Train GMM on train data
            gmm.fit(train.drop(TARGET, axis=1).values)
            if verbose:
                print('GMM Score for train set')
                print(gmm.score(train.drop(TARGET, axis=1).values))
                print('GMM Score for normal transcation subset in validation set')
                print(gmm.score(valid[valid[TARGET] == 0].drop(TARGET, axis=1).values))
                print('GMM Score for fraud transcation subset in validation set')
                print(gmm.score(valid[valid[TARGET] == 1].drop(TARGET, axis=1).values))

            # Figuring out a threshold range based on GMM score obtained from previous step
            tresholds = np.linspace(-800, 0, 100)
            # gmm.score_samples to calculate a GMM score for each data sample
            y_scores = gmm.score_samples(valid.drop(TARGET, axis=1).values)
            scores = []
            for treshold in tresholds:
                y_hat = (y_scores < treshold).astype(int)
                scores.append([recall_score(y_pred=y_hat, y_true=valid[TARGET].values),
                               precision_score(y_pred=y_hat, y_true=valid[TARGET].values),
                               f1_score(y_pred=y_hat, y_true=valid[TARGET].values)])

            scores = np.array(scores)
            if verbose:
                print(scores[:, 2].max(), scores[:, 2].argmax())

            final_tresh = tresholds[scores[:, 2].argmax()]
            if verbose:
                print('The final threshold selected is: ', final_tresh)

            y_hat_test = (gmm.score_samples(test.drop(TARGET, axis=1).values) < final_tresh).astype(int)

            # TO-DO: Call defined evaluation functions
            print('Final threshold: %f' % final_tresh)
            print('Test Recall Score: %.3f' % recall_score(y_pred=y_hat_test, y_true=test[TARGET].values))
            print('Test Precision Score: %.3f' % precision_score(y_pred=y_hat_test, y_true=test[TARGET].values))
            print('Test F1 Score: %.3f' % f1_score(y_pred=y_hat_test, y_true=test[TARGET].values))

            cnf_matrix = confusion_matrix(test[TARGET].values, y_hat_test)
            print("tn, fp, fn, tp:", cnf_matrix.ravel())

    # def check_shapes(self):
    #     print('Train shape: ', self.train.shape)
    #     print('Proportion os anomaly in training set: %.3f\n' % self.train[TARGET].mean())
    #     print('Valid shape: ', self.valid.shape)
    #     print('Proportion os anomaly in validation set: %.3f\n' % self.valid[TARGET].mean())
    #     print('Test shape:, ', self.test.shape)
    #     print('Proportion os anomaly in test set: %.3f\n' % self.test[TARGET].mean())
    #
    # def check_scores(self, gmm):
    #     print('GMM Score for train set')
    #     print(gmm.score(self.train.drop(TARGET, axis=1).values))
    #     print('GMM Score for normal transcation subset in validation set')
    #     print(gmm.score(self.valid[self.valid[TARGET] == 0].drop(TARGET, axis=1).values))
    #     print('GMM Score for fraud transcation subset in validation set')
    #     print(gmm.score(self.valid[self.valid[TARGET] == 1].drop(TARGET, axis=1).values))