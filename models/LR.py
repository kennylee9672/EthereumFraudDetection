import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, plot_confusion_matrix, precision_recall_curve, f1_score
from sklearn.metrics import precision_score, recall_score, auc, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from utils.split import kfold
from config import *


class Logistic_Regression_Model:
    def __init__(self, df):
        self.df = df
        self.model_name = 'Logistic Regression'
        self.preprocess_data()

    def update_dataframe(self, df):
        self.df = df
        self.preprocess_data()

    def get_model_name(self):
        return self.model_name

    def preprocess_data(self, test_size=None):
        self.df = shuffle(self.df)
        self.X = self.df.drop(columns=[TARGET], axis=1)
        self.y = self.df[TARGET]

    def run(self):
        print('Model Name: ', self.get_model_name())
        X, y = self.X, self.y

        f1_lbfgs = []
        accuracy_scores = []
        aurpc_scores = []
        precision_scores = []
        recall_scores = []

        for count, (train, test) in enumerate(kfold(X, y, 5)):
            print('Fold: ' + str(count))
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]
            
            
            lr = LogisticRegression(solver="saga")
            clf = lr.fit(X_train, y_train)
            pred = clf.predict(X_test)

            accuracy = accuracy_score(y_test, pred)
            print("Accuracy Score: ", accuracy)

            y_pred = clf.predict_proba(X_test)

            fraud_precision, fraud_recall, thresholds = precision_recall_curve(y_test, y_pred[:, 1])
            print("AUPRC:", auc(fraud_recall, fraud_precision))
            print("F1 score: ", f1_score(y_pred=pred, y_true=y_test), '\n')
            f1_lbfgs.append(f1_score(y_pred=pred, y_true=y_test))
            accuracy_scores.append(accuracy)
            aurpc_scores.append(auc(fraud_recall, fraud_precision))
            precision_scores.append(precision_score(y_pred=pred, y_true=y_test))
            recall_scores.append(recall_score(y_pred=pred, y_true=y_test))

            label = 'Fold: ' + str(count)
            plt.plot(fraud_recall, fraud_precision, label=label)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()
        mean_f1_lbfgs = sum(f1_lbfgs) / len(f1_lbfgs)
        mean_accuracy = sum(accuracy_scores)/ len(accuracy_scores)
        mean_aurpc = sum(aurpc_scores) / len(aurpc_scores)
        mean_precision_scores = sum(precision_scores)/ len(precision_scores)
        mean_recall_scores = sum(recall_scores)/len(recall_scores)
        
        print('Token Counts accuracy score: ', str(mean_accuracy))
        print('Token Counts precision score: ', str(mean_precision_scores))
        print('Token Counts recall score: ', str(mean_recall_scores))
        print('Token Counts f1 score lbfgs: ', str(mean_f1_lbfgs))
        print('Token Counts aurpc score: ', str(mean_aurpc))
        