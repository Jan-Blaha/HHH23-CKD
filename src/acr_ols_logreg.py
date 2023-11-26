import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


def fit_model(save=False):
    # open the data
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "honza_jirka_data.csv", delimiter=",")

    # keep relevant columns
    columns_to_keep = df.filter(regex='^(d|x|y)').columns
    df = df[columns_to_keep]
    df = df.reindex(sorted(df.columns), axis=1)

    # drop y_KREA
    df = df.drop('y_KREA', axis=1)

    # filter out the NaNs
    df.dropna(how='any', axis=0, inplace=True)
    df["x_sex"] = df["x_sex"].map({'M':0, 'F':1})

    # TODO throw out bottom and top 5%

    # train the model
    y = df["y_ACR"]
    X = sm.add_constant(df.drop('y_ACR', axis=1).astype(float))
    mod = sm.OLS(y, X)
    result = mod.fit()

    '''p_threshs = np.linspace(0., 0.2, 40)
    for p_thresh in p_threshs:
        high_p_cols = list(result.pvalues[result.pvalues > p_thresh].index)
        if 'const' in high_p_cols: high_p_cols.remove('const')
        _df = df.drop(columns=high_p_cols, axis=1)
        
        y = _df["y_ACR"]
        X = sm.add_constant(_df.drop('y_ACR', axis=1).astype(float))

        res = sm.OLS(y, X).fit()
        print(round(p_thresh, 3), res.aic)'''

    # throw out parameters that have high P-values and drop
    p_thresh = 0.1
    high_p_cols = list(result.pvalues[result.pvalues > 0.1].index)
    if 'const' in high_p_cols: high_p_cols.remove('const')
    df = df.drop(columns=high_p_cols, axis=1)

    # retrain the model
    y = df["y_ACR"]
    X = sm.add_constant(df.drop('y_ACR', axis=1).astype(float))
    mod = sm.OLS(y, X)
    result = mod.fit()

    # sort by the abs coef values
    sorted_coefficients = result.params.abs().sort_values()[::-1]
    X_sorted = X[sorted_coefficients.index]
    model_sorted = sm.OLS(y, X_sorted)
    result_sorted = model_sorted.fit()
    print(result_sorted.summary())

    if save:
        with open('acr_ols.pkl', 'wb') as f:
            pickle.dump(result_sorted, f)


def predict(data, model='acr_ols.pkl'):

    data["x_sex"] = data["x_sex"].map({'M':0, 'F':1})

    with open(model, 'rb') as f:
        loaded_model = pickle.load(f)

    # get the right columns and in order
    cols = list(loaded_model.params.index)
    X = data[cols]
    if X.isna().any().any():
        return None
    X = X.astype(float)

    # predict and get the pred interval
    pred = loaded_model.get_prediction(X)
          
    return pred.predicted_mean, pred.conf_int().squeeze()


def fit_log_reg(save=False):
    # open the data
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "honza_jirka_data.csv", delimiter=",")

    # keep relevant columns
    columns_to_keep = df.filter(regex='^(d|x|y)').columns
    df = df[columns_to_keep]
    df = df.reindex(sorted(df.columns), axis=1)

    # drop y_KREA, map x_sex
    df = df.drop('y_KREA', axis=1)
    df["x_sex"] = df["x_sex"].map({'M':0, 'F':1})

    # get label and drop
    y = np.array(df["y_ACR"].isna().astype(int))
    df = df.drop("y_ACR", axis=1)

    # get indices of rows with valid features
    indices = df[df.notna().all(axis=1)].index
    y = y[indices]
    X = df.loc[indices].astype(float)

    # train model on whole data
    X = sm.add_constant(X)
    logreg_model = sm.Logit(y, X)
    result = logreg_model.fit(disp=True, method='lbfgs')
    print(result.summary())

    # save model and the columns
    if save:
        with open('logreg.pkl', 'wb') as f:
            pickle.dump(logreg_model, f)


if __name__ == "__main__":
    #df = pd.read_csv(Path(__file__).parent.parent / "data" / "honza_jirka_data.csv", delimiter=",")
    #df = sm.add_constant(df)
    #predict(df.iloc[[2111]])

    # fit logreg
    fit_log_reg(save=True)
