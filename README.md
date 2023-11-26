# Solution to the European Healthcare Hackathon 2023 Challenge 6 - Estimating Risk of CKD in Patients

by team Zdravíme


## Outcomes

- [IKEM prevalence data](presentation/ikem_prevalence_results.txt)
- High ACR risk predictive model


## Solution overview
Based on the medical records, we can give a doctor the risk of a particular patient being in the A2 and A3 CKD risk categories.
Because the albuminuria is not standardly tested, we can recommend such action to the doctor when actually needed!

![image](https://github.com/Jan-Blaha/HHH23-CKD/assets/36329222/b229c7c5-9e63-4dfe-899f-84f06574d69b)


## Structure of the repository:
- data/ - placeholder
- src/ - our solution
- presentation/ - some outcomes of our work


## Used technologies:
- we developed our solutions in Python Jupyter notebooks and Julia.
- from analytical tools, we relied mostly on statsmodels library and our wit; we also tested causal inference tools and advanced ideas like sum-product-networks for sum-clever-analysis.


## Technical detail:
Our final solution builds on a two-level approach.
First, by estimating the availability of ACR testing results based on other covariates, we understand the sampling bias of the data.
Then, we can properly learn a model predicting the ACR levels for a particular patient and adjusting by inverse probability weighting for the sampling bias.
Given the estimated ACR levels and their uncertainty, we give the doctor a percentual risk of the patient being above the defined thresholds of the A2 and A3 CKD-albuminuria-based category.




