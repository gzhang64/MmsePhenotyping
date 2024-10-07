from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression



def initialize_feature_selection(k, train_x_raw, train_y, val_x_raw):
    feature_selector = SelectKBest(score_func=mutual_info_regression, k=k)
    feature_selector.fit(train_x_raw, train_y)
    train_x = feature_selector.transform(train_x_raw)
    val_x = feature_selector.transform(val_x_raw)
    return train_x, val_x



if __name__ == "__main__":
    pass