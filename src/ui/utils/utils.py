# Utility functions

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Part Worth 
def plot_part_worth(conjoint_df):
    # Logit Model for Conjoint Analysis
    # Data Format
    y = conjoint_df['response']
    X = conjoint_df.iloc[:,1:]
    # Model
    res = sm.OLS(y, X, family=sm.families.Binomial()).fit()
    # Tabulating Results
    df_res = pd.DataFrame({
        'param_name': res.params.keys()
        , 'param_w': res.params.values
        , 'pval': res.pvalues
    })
    # adding field for absolute of parameters
    df_res['abs_param_w'] = np.abs(df_res['param_w'])
    # marking field is significant under 95% confidence interval
    df_res['is_sig_95'] = (df_res['pval'] < 0.065)
    # constructing color naming for each param
    df_res['c'] = ['blue' if x else 'red' for x in df_res['is_sig_95']]
    # make it sorted by abs of parameter value
    df_res = df_res.sort_values(by='abs_param_w', ascending=True)
    # Visualising Results
    fig, ax = plt.subplots(figsize=(14, 12))
    plt.title('Part Worth')
    pwu = df_res['param_w']
    xbar = np.arange(len(pwu))
    plt.barh(xbar, pwu, color=df_res['c'])
    plt.yticks(xbar, labels=df_res['param_name'])
    return fig, res

# Feature Importance
def plot_feature_importance(res):
    # Calculating Absolute Attribute Importance
    # need to assemble per attribute for every level of that attribute in dicionary
    range_per_feature = dict()
    for key, coeff in res.params.items():
        sk =  key.split('_')
        feature = sk[0]
        if len(sk) == 1:
            feature = key
        if feature not in range_per_feature:
            range_per_feature[feature] = list()
            
        range_per_feature[feature].append(coeff)

    # importance per feature is range of coef in a feature
    # while range is simply max(x) - min(x)
    importance_per_feature = {
        k: max(v) - min(v) for k, v in range_per_feature.items()
    }

    # compute relative importance per feature or normalized feature importance by dividing sum of importance for all features
    total_feature_importance = sum(importance_per_feature.values())
    relative_importance_per_feature = {
        k: 100 * round(v/total_feature_importance, 3) for k, v in importance_per_feature.items()
    }
    # Tabulating Feature Importance
    alt_data = pd.DataFrame(
        list(importance_per_feature.items()), 
        columns=['attr', 'importance']
    ).sort_values(by='importance', ascending=True)

    # Calculating Relative Percentage Attribute Importance
    alt_data = pd.DataFrame(
        list(relative_importance_per_feature.items()), 
        columns=['attr', 'relative_importance (pct)']
    ).sort_values(by='relative_importance (pct)', ascending=True)

    # Visualising Relative Percentage Attribute Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    xbar = np.arange(len(alt_data['attr']))
    plt.title('Relative importance / Normalized importance')
    plt.barh(xbar, alt_data['relative_importance (pct)'])
    for i, v in enumerate(alt_data['relative_importance (pct)']):
        ax.text(v + 0.2, i  , '{:.2f}%'.format(v))
    plt.ylabel('attributes')
    plt.xlabel('% relative feature importance')
    plt.yticks(xbar, alt_data['attr'])
    plt.box(on = False)
    return fig