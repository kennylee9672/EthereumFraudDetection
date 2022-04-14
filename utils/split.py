from sklearn.model_selection import KFold

def kfold(X, y, n_splits, test_size=0.7):
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(X, y)
    
    return kf.split(X)