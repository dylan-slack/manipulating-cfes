import numpy as np
from sklearn.datasets import make_blobs
import utils_config
import pandas as pd

config_file_d="./conf/datasets.json"
config_d = utils_config.load_config(config_file_d)
config_d = utils_config.serialize_config(config_d)

PROTECTED = config_d['PROTECTED']
NOT_PROTECTED = config_d['NOT_PROTECTED']
POSITIVE = config_d['POSITIVE']
NEGATIVE = config_d['NEGATIVE']

def get_and_preprocess_cc():
    """"Handle processing of Communities and Crime.  We exclude rows with missing values and predict
    if the violent crime is in the 50th percentile.
    Parameters
    ----------
    params : Params
    Returns:
    ----------
    Pandas data frame X of processed data, np.ndarray y, and list of column names
    """

    X = pd.read_csv("./data/communities_and_crime.csv", index_col=0)
    
    # everything over 50th percentil gets negative outcome (lots of crime is bad)
    high_violent_crimes_threshold = 50
    y_col = 'ViolentCrimesPerPop numeric'

    X = X[X[y_col] != "?"]
    X[y_col] = X[y_col].values.astype('float32')

    # just dump all x's that have missing values 
    cols_with_missing_values = []
    for col in X:
        if '?' in X[col].values.tolist():
            cols_with_missing_values.append(col)    

    y = X[y_col]
    y_cutoff = np.percentile(y, high_violent_crimes_threshold)

    protected_col = X['racepctblack numeric']
    protected_attribute = np.array([PROTECTED if val > np.percentile(protected_col, 50) else NOT_PROTECTED for val in protected_col])

    X = X.drop(cols_with_missing_values + ['racepctblack numeric', 'communityname string', 'fold numeric', 'county numeric', 'community numeric', 'state numeric'] + [y_col], axis=1)

    # setup ys
    y = np.array([NEGATIVE if val > y_cutoff else POSITIVE for val in y])
    return X.values, y, protected_attribute, np.array([]) #np.array([17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 74, 34, 72, 80, 56, 13, 23, 84, 40, 68,1,2,4,5,6,7,8,9,10,11,12,14,15,16])

def get_and_preprocess_german():
    X = pd.read_csv("data/german_processed.csv")
    y = X["GoodCustomer"]

    X = X.drop(["GoodCustomer", "PurposeOfLoan"], axis=1)
    X['Gender'] = [NOT_PROTECTED if v == "Male" else PROTECTED for v in X['Gender'].values]

    y = np.array([POSITIVE if p == 1 else NEGATIVE for p in y.values])

    p = X['Gender'].values
    X = X.drop(['Gender'], axis=1)

    num = [2,3,4,5,6,7,8]
    cat = [v for v in range(X.shape[1]) if v not in num]

    cols = [c for c in X]

    return X.values, y, p, np.array(cat)

def get_and_preprocess_adult():

    from dice_ml.utils import helpers

    data = helpers.load_adult_income_dataset()

    y = data['income'].values
    p = data['gender'].values
    p = np.array([PROTECTED if v == 'Female' else NOT_PROTECTED for v in p])

    data = data.drop(['income', 'gender'], axis=1)
    data = pd.get_dummies(data)
    X = data.values

    cols = [c for c in data]
    cat = [i for i, c in enumerate(cols) if c not in ['age', 'hours_per_week']]

    return X, y, p, np.array(cat)

def get_and_preprocess_compas_data():
    """Handle processing of COMPAS according to: https://github.com/propublica/compas-analysis
    
    Parameters
    ----------
    params : Params
    Returns
    ----------
    Pandas data frame X of processed data, np.ndarray y, and list of column names
    """

    compas_df = pd.read_csv("data/compas-scores-two-years.csv", index_col=0)
    compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
                              (compas_df['days_b_screening_arrest'] >= -30) &
                              (compas_df['is_recid'] != -1) &
                              (compas_df['c_charge_degree'] != "O") &
                              (compas_df['score_text'] != "NA")]

    compas_df['length_of_stay'] = (pd.to_datetime(compas_df['c_jail_out']) - pd.to_datetime(compas_df['c_jail_in'])).dt.days
    # X = compas_df[['age', 'c_charge_degree', 'race', 'sex', 'priors_count', 'length_of_stay']]
    X = compas_df[['age', 'race', 'priors_count', 'length_of_stay']]

    # print (compas_df['score_text'])

    # if person has high score give them the _negative_ model outcome
    y = np.array([NEGATIVE if score == 'High' else POSITIVE for score in compas_df['score_text']])
    sens = X.pop('race')

    # assign African-American as the protected class
    X = pd.get_dummies(X)
   
    sensitive_attr = np.array(pd.get_dummies(sens).pop('African-American'))
    sensitive_attr = np.array([PROTECTED if val == 1 else NOT_PROTECTED for val in sensitive_attr])

    return X.values, y, sensitive_attr

def get_data_set(name):
    """
    Gets the data set given the name.  If sythetic, creates blobs sythetic dataset example.
    """
    if name == "synthetic":
        data, t_labels = make_blobs(n_samples=100, n_features=2, cluster_std=2, centers=[[8,5], [3,1], [-4,1], [-8,5]])
        labels = np.zeros_like(t_labels)
        labels[np.logical_or(t_labels == 0, t_labels == 1)] = POSITIVE
        labels[np.logical_or(t_labels == 2, t_labels == 3)] = NEGATIVE

        protected = np.zeros_like(t_labels)
        protected[np.logical_or(t_labels == 0, t_labels == 3)] = PROTECTED
        protected[np.logical_or(t_labels == 1, t_labels == 2)] = NOT_PROTECTED
        cats = np.array([])
    elif name == "compas":
        data, labels, protected = get_and_preprocess_compas_data()
    elif name == "cc":
        data, labels, protected, cats = get_and_preprocess_cc()
    elif name == "german":
        data, labels, protected, cats = get_and_preprocess_german()
    elif name == "adult":
        data, labels, protected, cats = get_and_preprocess_adult()
    else:
        raise NotImplementedError

    training = np.random.choice(data.shape[0], size=int(data.shape[0] * 0.9)) 
    testing = np.array([i for i in range(data.shape[0]) if i not in training])

    return data[training], labels[training], protected[training], data[testing], labels[testing], protected[testing], cats
