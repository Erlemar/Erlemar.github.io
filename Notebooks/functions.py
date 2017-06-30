import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import _tree



def feature_stat(data_, feature, target_name):
    data = data_.copy()
    print('Counts:')
    print(data.groupby(feature)[target_name].count())
    print('Frequencies:')
    print(data[feature].value_counts(normalize=True, dropna=False))
    x = [i for i in data.groupby(feature)[target_name].count().index]
    
    if data[feature].isnull().any():
        if str(data[feature].dtype) == 'category':
            data[feature].cat.add_categories(['None'], inplace=True)
        data[feature].fillna('None', inplace=True)
        
    y1 = [i for i in data.groupby(feature)[target_name].count().values]
    y2 = [i for i in data.groupby(feature)[target_name].mean().values]
    ind = np.arange(len(data[feature].unique()))
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(ind, y1, align='center', width=0.4, alpha=0.7)
    ax1.set_xlabel(feature)
    ax1.set_ylabel('Counts', color='b')
    ax1.tick_params('y1', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(ind, y2, 'r')
    ax2.set_ylabel('Mean rate', color='r')
    ax2.tick_params('y2', colors='r')
    plt.xticks(ind, x, rotation=45)
    ax1.set_xticklabels(x, rotation=35)
    plt.grid(False)
    plt.show()
    _, iv = calc_iv(data, target_name, feature)
    print('IV: ', iv)
    
def cont_split(data, feature, target, leafs, auto_calc=False):
    x = data[feature]
    y = data[target]
    
    clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=0.05, min_samples_leaf=0.05, max_features=1, min_weight_fraction_leaf=0.1,  max_leaf_nodes=leafs, class_weight='balanced')
    clf = clf.fit(x.values.reshape(-1, 1), y)
    
    h = tree_to_thresholds(clf, 'a')
    h = list(h)
    h.append(max(x))
    if feature == 'AGE':
        d = {k: np.round(v, 0) for k,v in zip(range(len(h)),h)}
    elif data[feature].max() > 1000:
        d = {k: np.round(v, -2) for k,v in zip(range(len(h)),h)}
    else:
        d = {k: v for k,v in zip(range(len(h)),h)}
        
    if len(set(d.values())) < len(list(d.values())):
        return None
    
    if auto_calc == True:
        temp_series = pd.DataFrame({feature: pd.cut(data[feature], bins=[v for v in d.values()]), target: data[target]})
        temp_series[feature].cat.add_categories(['No value'], inplace=True)
        return calc_iv(temp_series, target, feature)[1]
    
    #data.groupby(pd.cut(x, bins=[v for v in d.values()]))[target].mean().plot(kind='bar')
    temp_series = pd.DataFrame({feature: pd.cut(data[feature], bins=[v for v in d.values()]), target: data[target]})
    temp_series[feature].cat.add_categories(['No value'], inplace=True)

    print(pd.cut(x, bins=[v for v in d.values()]).value_counts(normalize=True, dropna=False))

    print('IV: ', calc_iv(temp_series, target, feature)[1])
        
    return pd.cut(data[feature], bins=[v for v in d.values()])


def tree_to_thresholds(tree, feature_names):
    '''
    Get thresholds from decision tree.
    '''
    
    tr = [0]
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if threshold not in tr: tr.append(threshold)
            recurse(tree_.children_left[node], depth + 1)
            if threshold not in tr: tr.append(threshold)
            recurse(tree_.children_right[node], depth + 1)
        else:
            pass

    recurse(0, 1)
    return sorted(tr)

def calc_iv(data, target, feature):
    df = pd.DataFrame(index = data[feature].unique(),
                      data={'% responders': data.groupby(feature)[target].sum() / np.sum(data[target])})
    df['% non-responders'] = (data.groupby(feature)[target].count() - data.groupby(feature)[target].sum()) / (len(data[target]) - np.sum(data[target]))
    df['WOE'] = np.log(df['% responders'] / df['% non-responders'])
    df['DG-DB'] = df['% responders'] - df['% non-responders']
    df['IV'] = df['WOE'] * df['DG-DB']
    return df, np.sum(df['IV'])

def split_best_iv(data, feature, target_name):
    best_iv = 0
    for i in range(2, 20):
        iv_temp = cont_split(data, feature, target_name, i, auto_calc=True)
        if iv_temp == None:
            return cont_split(data, feature, target_name, i - 1)
        if iv_temp > best_iv:
            best_iv = iv_temp
        else:
            return cont_split(data, feature, target_name, i - 1)