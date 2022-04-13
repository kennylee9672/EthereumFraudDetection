from typing import Counter
import imblearn

def smote(X, y, random_state=None, verbose=False):
    """
    SMOTE: Synthetic Minority Over-sampling Technique

    https://imbalanced-learn.org/stable/
    """
    if verbose:
        print('Original Data Distribution: ', Counter(y))
    sm = imblearn.over_sampling.SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    if verbose:
        print('Data Distribution After SMOTE: ', Counter(y_res))
    return X_res, y_res

