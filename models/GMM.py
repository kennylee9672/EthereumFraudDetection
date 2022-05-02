import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, plot_confusion_matrix, precision_recall_curve, f1_score
from sklearn.metrics import precision_score, recall_score, auc, accuracy_score
from sklearn.mixture import GaussianMixture

from config import *

from sklearn.mixture import GaussianMixture

class GMM_Model():
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
        # self.check_shapes()
        
        return (train, valid, test)
    
    def compute_threshold_range(self, model, train, valid):
        x1 = model.score(train.drop(TARGET, axis=1).values)
        x2 = model.score(valid[valid[TARGET] == 0].drop(TARGET, axis=1).values)
        x3 = model.score(valid[valid[TARGET] == 1].drop(TARGET, axis=1).values)
        l = sorted([x1, x2, x3])
        
        return (min(l), max(l))

    def run(self, verbose=False):
        train, valid, test = self.train, self.valid, self.test
        for components in range(1, 4):
            gmm = GaussianMixture(n_components=components, n_init=4, random_state=42, covariance_type='full')
            gmm.fit(train.drop(TARGET, axis=1).values)

            # Compute thresholds
            threshold_range = self.compute_threshold_range(gmm, train, valid)
            thresholds = np.linspace(threshold_range[0], threshold_range[1], 100)
            print('Threshold range: %.3f, %.3f' % (threshold_range[0], threshold_range[1]))
            
            # Calculate score for each data sample
            scores = []
            for th in thresholds:
                y_pred_proba = gmm.score_samples(valid.drop(TARGET, axis=1).values)
                y_hat = y_pred_proba.copy()
                y_valid = valid[TARGET].values
                
                y_hat[y_hat >= th] = 0
                y_hat[y_hat < th] = 1
                scores.append([recall_score(y_hat, y_valid),
                               precision_score(y_hat, y_valid),
                               f1_score(y_hat, y_valid)])
            scores = np.array(scores)
            
            # Plot Result
            self.plot_evaluations(components, thresholds, scores)

            # Evaluate on test data with optimal threshold
            final_threshold = thresholds[scores[:, 2].argmax()]
            y_hat = self.evaluate_test(gmm, final_threshold)
            y_test = test[TARGET].values
            
            # Output results
            if verbose:
                self.check_scores(gmm)
                print(classification_report(y_test, y_hat, target_names=CLASSES))

    def evaluate_test(self, model, threshold):
        X_test = self.test.drop(TARGET, axis=1).values
        y_test = self.test[TARGET]
        
        # get predictions
        y_pred_proba = model.score_samples(X_test)
        y_hat = y_pred_proba.copy()

        # check prediction accuracy based on such threshold
        y_hat[y_hat >= threshold] = 0
        y_hat[y_hat < threshold] = 1

        recall = recall_score(y_test, y_hat)
        precision = precision_score(y_test, y_hat)
        f1 = f1_score(y_test, y_hat)
        accuracy = accuracy_score(y_test, y_hat)
        cf = confusion_matrix(y_test, y_hat)

        print('Optimal threshold: %.3f' % threshold)
        print('Recall score: %.3f' % recall)
        print('Precision score: %.3f' % precision)
        print('F1 Score: %.3f' % f1)
        print('Accuracy: %.3f' % accuracy)
        
        return y_hat
            
    def plot_evaluations(self, n, thresholds, scores):            
        fig, ax = plt.subplots()
        score_names = ['Recall', 'Precesion', 'F1']
        for i in range(len(score_names)):
            plt.plot(thresholds, scores[:, i])
        ax.legend(score_names)
        plt.title('GMM\'s %s components' % str(n))
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.show()

    def check_shapes(self):
        print('Train shape: ', self.train.shape)
        print('Proportion os anomaly in training set: %.3f' % self.train[TARGET].mean())
        print('Valid shape: ', self.valid.shape)
        print('Proportion os anomaly in validation set: %.3f' % self.valid[TARGET].mean())
        print('Test shape:, ', self.test.shape)
        print('Proportion os anomaly in test set: %.3f\n' % self.test[TARGET].mean())
    
    def check_scores(self, model):
        print('GMM Score for train set: %.3f' % model.score(self.train.drop(TARGET, axis=1).values))
        print('GMM Score for normal transcation in validation set: %.3f' % model.score(self.valid[self.valid[TARGET] == 0].drop(TARGET, axis=1).values))
        print('GMM Score for fraud transcation in validation set: %.3f\n' % model.score(self.valid[self.valid[TARGET] == 1].drop(TARGET, axis=1).values))