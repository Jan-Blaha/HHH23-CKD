import pickle
import statsmodels 
import statsmodels.api as sm
from statsmodels.treatment.treatment_effects import TreatmentEffect
import scipy.stats
import pandas as pd
import numpy as np




with open("glm_acr.pkl", "rb") as f:
    result = pickle.load(f)

#names = result.model.exog.columns
coefficients = result.params
pvals = result.pvalues

df = pd.DataFrame(data={"coefficient": coefficients, "pval": pvals})

pval_th = 0.1

#coefficients = coefficients.loc[pvals < pval_th]
#coefficients_sorted = coefficients.iloc[np.argsort(coefficients.values)[::-1]]

df = df.loc[df.pval < pval_th]
df = df.sort_values("coefficient", ascending=False)

print(df.iloc[:20])

