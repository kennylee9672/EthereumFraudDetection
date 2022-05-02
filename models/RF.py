import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, plot_confusion_matrix, precision_recall_curve, f1_score
from sklearn.metrics import precision_score, recall_score, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_confusion_matrix

from config import *
from utils.split import kfold


class Random_Forest_Model:
    def __init__(self, df):
        self.df = df
        self.max_depth = 100
        self.test_size = 0.7
        self.model_name = 'RandomForest'
        self.batch = 5
        self.preprocess_data()

    def get_model_name(self):
        return self.model_name

    def set_max_depth(self, depth):
        self.max_depth = depth

    def set_test_size(self, size):
        self.test_size = size

    def set_batch(self, batch):
        self.batch = batch

    def preprocess_data(self):
        self.X = self.df.drop(columns=[TARGET])
        self.y = self.df[TARGET]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42)

    def run(self):
        depth = self.compute_optimal_depth()
        rfc = RandomForestClassifier(max_depth=depth)

        f1_lbfgs = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        aurpc_scores = []

        X, y = self.X, self.y
        for count, (train, test) in tqdm(enumerate(kfold(X, y, self.batch))):
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]

            rfc = rfc.fit(X_train, y_train)
            y_pred_proba = rfc.predict_proba(X_test)
            y_pred = rfc.predict(X_test)

            fraud_precision, fraud_recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
            f1_lbfgs.append(f1_score(y_pred=y_pred, y_true=y_test))
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)
            aurpc_scores.append(auc(fraud_recall, fraud_precision))
            precision_scores.append(precision_score(y_pred=y_pred, y_true=y_test))
            recall_scores.append(recall_score(y_pred=y_pred, y_true=y_test))
            print("AUPRC:", auc(fraud_recall, fraud_precision))
            print("F1 score: ", f1_score(y_pred=y_pred, y_true=y_test), '\n')

            self.evaluate(y_test, y_pred)
            self.plot_PRC(y_test, y_pred_proba, count)

        mean_f1 = sum(f1_lbfgs) / len(f1_lbfgs)
        mean_accuracy = sum(accuracy_scores)/ len(accuracy_scores)
        mean_aurpc = sum(aurpc_scores) / len(aurpc_scores)
        mean_precision = sum(precision_scores)/ len(precision_scores)
        mean_recall = sum(recall_scores)/len(recall_scores)
        print('Token Counts accuracy score: ', str(mean_accuracy))
        print('Token Counts precision score: ', str(mean_precision))
        print('Token Counts recall score: ', str(mean_recall))
        print('Token Counts f1 score: ', str(mean_f1))
        print('Token Counts aurpc score: ', str(mean_aurpc))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()

    def evaluate(self, y_test, y_pred):
        print(classification_report(y_test, y_pred, target_names=CLASSES))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        print('Confusion Matrix: [tn, fp, fn, tp] = ', cm.ravel())
        # fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True, class_names=CLASSES)
        # plt.show()

    def plot_PRC(self, y_test, y_pred, i):
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred[:, 1])
        plt.plot(precision, recall, label='Fold: ' + str(i))

    def compute_optimal_depth(self):
        accuracy_rf, max_depth_rf = [], []
        for i in range(1, self.max_depth):
            clf_rfr = RandomForestClassifier(max_depth=i)
            clf_rfr = clf_rfr.fit(self.X_train, self.y_train)
            y_pred = clf_rfr.predict(self.X_test)

            accuracy_rf.append(accuracy_score(self.y_test, y_pred))
            max_depth_rf.append(i)
        # plt.plot(max_depth_rf, accuracy_rf)
        # plt.show()

        optimal_index = np.array(accuracy_rf).argmax()
        optimal_depth = max_depth_rf[optimal_index]
        print('Highest Accuracy At Depth: ', optimal_depth)

        return optimal_depth
