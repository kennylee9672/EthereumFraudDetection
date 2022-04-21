from sklearn.linear_model import LogisticRegression
from utils.split import kfold


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

        for count, (train, test) in enumerate(kfold(X, y, 5)):
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

            label = 'Fold: ' + str(count)
            plt.plot(fraud_recall, fraud_precision, label=label)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()
        mean_f1_lbfgs = sum(f1_lbfgs) / len(f1_lbfgs)
        print('Token Counts f1 score lbfgs: ', str(mean_f1_lbfgs))