#테스트 용도 코드 (github 잘 올라가는지)
#iris data load해서, catboost default parameter 이용하여 (except for depth) 
#test stat 어떻게 나오는지 확인해보기.

import pandas as pd

from sklearn.datasets import load_iris
import catboost as cb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

iris_data = load_iris()

df_X = pd.DataFrame(data=iris_data["data"], columns=iris_data["feature_names"])
df_y = pd.DataFrame(data=iris_data["target"], columns=["y"])

df_Agg = pd.concat([df_X, df_y])


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(skf.split(df_X, df_y)):
    df_X_train, df_y_train = df_X.loc[train_index], df_y.loc[train_index]
    df_X_test, df_y_test = df_X.loc[test_index], df_y.loc[test_index]
    mdl = cb.CatBoostClassifier(max_depth=2, verbose=False)
    mdl.fit(df_X_train, df_y_train)
    df_yhat_test = mdl.predict(df_X_test)
    df_yprob_test = mdl.predict_proba(df_X_test)
    print(confusion_matrix(df_y_test, df_yhat_test))
    print(classification_report(df_y_test, df_yhat_test))
    print(roc_auc_score(df_y_test, df_yprob_test, multi_class="ovr"))


