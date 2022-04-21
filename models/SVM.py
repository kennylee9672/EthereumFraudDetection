from sklearn import svm
from utils.split import kfold


class SVM_Model:
    def __init__(self, df):
        self.df = df
        self.model_name = 'SVM'
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

        f1_scores = []

        for count, (train, test) in enumerate(kfold(X, y, 5)):
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]

            clf = svm.SVC(probability=True)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)
            y_hat = clf.predict(X_test)
            # Model Accuracy, how often is the classifier correct?
            print("Accuracy:", accuracy_score(y_test, y_hat))
            print("F1 score:", f1_score(y_test, y_hat))
            f1_scores.append(f1_score(y_test, y_hat))
            fraud_precision, fraud_recall, thresholds = precision_recall_curve(y_test, y_pred[:, 1])
            print("AUPRC:", auc(fraud_recall, fraud_precision))
            # plot the precision-recall curves
            no_skill = len(y_test[y_test == 1]) / len(y_test)
            label = 'Fold: ' + str(count)
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label=label + ' No Skill')
            plt.plot(fraud_recall, fraud_precision, marker='.', label=label + ' Support Vector Machine')

        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()
        mean_f1_scores = sum(f1_scores) / len(f1_scores)
        print('Token Counts f1 score lbfgs: ', str(mean_f1_scores))