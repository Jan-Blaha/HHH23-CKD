import pickle
import statsmodels 
import statsmodels.api as sm
from statsmodels.treatment.treatment_effects import TreatmentEffect
import scipy.stats
import pandas as pd
import numpy as np


print("Read data")
#df = pd.read_csv("honza_jirka_data.csv", delimiter=",")
#df.to_parquet("cache.parquet")

df = pd.read_parquet("cache.parquet")


# keep relevant columns
columns_to_keep = df.filter(regex='^(d|x|y)').columns
df = df[columns_to_keep]
df = df.reindex(sorted(df.columns), axis=1)

# drop y_KREA, map x_sex
df = df.drop('y_KREA', axis=1)
df["x_sex"] = df["x_sex"].map({'M':0, 'F':1})

# get label and drop
y = (~(df["y_ACR"].isna())).astype(int)

logreg_acr_values = df["y_ACR"]

df = df.drop("y_ACR", axis=1)

# get indices of rows with valid features
indices = df[df.notna().all(axis=1)].index
y = y.loc[indices]
X = df.loc[indices].astype(float)

logreg_acr_values = logreg_acr_values.loc[indices]

X = sm.add_constant(X)


print("Train LogReg")

try:

    with open("logreg.pkl", "rb") as f:
        result = pickle.load(f)
except:
    # train model on whole data
    logreg_model = sm.Logit(y, X)
    max_iter = 100
    result = logreg_model.fit(disp=True, method='lbfgs', maxiter=max_iter)

print(result.summary())

# save model and the columns
#with open('logreg.pkl', 'wb') as f:
#    pickle.dump(result, f)
result.save('logreg.pkl')


logreg_X, logreg_y = X, y



df = pd.read_parquet("cache.parquet")

# keep relevant columns
columns_to_keep = df.filter(regex='^(d|x|y)').columns
df = df[columns_to_keep]
df = df.reindex(sorted(df.columns), axis=1)

# drop y_KREA
df = df.drop('y_KREA', axis=1)

# filter out the NaNs
df.dropna(how='any', axis=0, inplace=True)
df["x_sex"] = df["x_sex"].map({'M':0, 'F':1})

# throw out bottom and top 5%
bottom_percentile = df["y_ACR"].quantile(0.05)
top_percentile = df["y_ACR"].quantile(0.95)
df = df.sort_values(by="y_ACR")
df = df[(df["y_ACR"] > bottom_percentile) & (df["y_ACR"] < top_percentile)]

y = df["y_ACR"]
X = sm.add_constant(df.drop('y_ACR', axis=1).astype(float))

print("Train GLM")

test_pred_model = result
# p = 0.3
# ind = np.random.choice([True, False], p=(p, 1-p), size=len(X))
ind = np.ones_like(y, dtype=bool)
X_sub = X.loc[ind]
y_sub = y.loc[ind]


#test_pred_model.model.exog = test_pred_model.model.exog[~np.isnan(test_pred_model.model.exog)]  # .dropna(how='any', axis=0)

test_pred_model.model.exog = X

try:
    with open("glm_acr.pkl", "rb") as f:
        result = pickle.load(f)
except:
    mod = sm.GLM(y_sub, X_sub, freq_weights=(1/test_pred_model.predict())[ind])
    result = mod.fit()
    result.summary()




logreg_X
bottom_percentile = df["y_ACR"].quantile(0.05)
top_percentile = df["y_ACR"].quantile(0.95)
#df = df.sort_values(by="y_ACR")
ind = (df["y_ACR"] > bottom_percentile) & (df["y_ACR"] < top_percentile)
p = result.get_prediction(exog=logreg_X) # [(logreg_y > bottom_percentile) & (logreg_y < top_percentile) | pd.isna(logreg_y)])

p.summary_frame()


result.save('glm_acr.pkl')

p.summary_frame().to_csv("prediction_results.csv")



df = pd.read_parquet("cache.parquet")

def risk(pred_mean, pred_sd, risk_threshold):
    return 1 - scipy.stats.distributions.norm.cdf(risk_threshold, loc=pred_mean, scale=pred_sd)

def calculate_risks(data):
    risk_A2 = risk(data["mean"], data.mean_se, 3)
    risk_A3 = risk(data["mean"], data.mean_se, 30)
    return risk_A2, risk_A3

risk_A2, risk_A3 = calculate_risks(p.summary_frame())


monitored_indices = ~df["y_ACR"].isna()
#monitored_indices = ~pd.isna(y_sub)
tmp_df = df.loc[monitored_indices]
columns_to_check = ["d_N18", "d_N18.0", "d_N18.1", "d_N18.2", "d_N18.3", "d_N18.4", "d_N18.5", "d_N18.8", "d_N18.9"]
ckd_diagnosed = tmp_df[columns_to_check].any(axis=1)
monitored_nodiag = tmp_df.loc[~ckd_diagnosed]["y_ACR"] > 3
monitored_diag = np.logical_and(tmp_df["y_ACR"] > 3, ckd_diagnosed)
print(f"There is {tmp_df.shape[0]} patients WITH monitored values of ACR, out of those:")
print(f"- {np.sum(ckd_diagnosed)} have some form of CKD")
print(f"- {np.sum(monitored_nodiag)} have risk of having ACR without having the diagnose")
print(f"- {np.sum(monitored_diag)} have risk of having ACR and have diagnose of CKD")

# not-monitored
not_monitored_indices = logreg_y == 0
#tmp_df = df.loc[~monitored_indices]
risk_A2A3 = risk_A2[not_monitored_indices] > 0.5
risk_A3 = risk_A3[not_monitored_indices] > 0.5
print(f"There is {not_monitored_indices.sum()} patients WITHOUT monitored values of ACR, out of those:")
print(f"- {np.sum(risk_A2A3)} have HIGHER risk of having CKD")
print(f"- {np.sum(risk_A3)} have VERY HIGH risk of having CKD")

print(f"Model callibration to known data:")
print(f"Actual risk for people we flag (prob of ACR > 3 is estimated > .5) is {(logreg_acr_values.values[(logreg_y == 1) & (risk_A2 > .5)] > 3).mean():.3f}")



