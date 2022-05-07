from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd

def split_data(df, target, drop_list=None, scaler=None):
    '''
    When using scalers: \n
    'standard' scales around the distribution mean and standard deviation.  \\
    'minmax' transforms data to fall within 0 and 1.
    
    Parameters
    ----------
    df : df
        entire dataset, including target
    target : str
        target column name
    drop_list : str, list
        default=None | column names to drop
    scaler : str
        default=None | 'standard' or 'minmax' 
    
    Returns
    -------
    X_train : array
        dataset of dependent features to train
    X_test : array
        dataset of dependent features to test
    y_train : array
        target independent feature to train
    y_test : array
        target independent feature to evaluate predicitons against
    
    '''
    
    # import libraries
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    # optionally drop column, not target
    if drop_list == None:
        print('\nNo columns dropped.\n')
    else:
        df = df.drop(drop_list,axis=1)
        print(f'\nColumn(s) dropped: {drop_list}')
    
    # define target and drop from predictors
    y = df[target].values #target
    X = df.drop([target],axis=1).values #features
    print(f'Target values: {y}', '\n')
    print(f'Column(s) remaining: {df.columns}\n')
    
    # scale data using MinMax or Standard
    if scaler == None:
        print('Data is unscaled.')
    elif scaler == 'minmax':
        scl = MinMaxScaler()
        scl.fit(X)
        X = scl.transform(X)
    elif scaler == 'standard':
        scl = StandardScaler()
        scl.fit(X)
        X = scl.transform(X)
    
    # train-test split, random_state defined for repeatability
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("training sample size: ", len(y_train))
    print("testing sample size: ", len(y_test))
    # print("fraud cases in test-set: ", sum(y_test))
    
    return X_train, X_test, y_train, y_test


def get_predictions(clf, X_train, y_train, X_test):
    '''
    Pass a classifier class object along with the datasets to train and test. \\
    
    Parameters
    ----------
    clf : class
        str(lr) for LinearRegression() | classifier with optional parameters
    X_train : df
        predictor training data
    y_train : df
        target training data
    X_test : df
        evaluation data
    params : dict
        parameters to be passsed in classifier
    
    Returns
    -------
    y_pred
        predicted classes
    y_prob
        predicted probabilities 
    
    '''
    from sklearn.linear_model import LinearRegression
    
    if clf == 'lr':
        estimator = LinearRegression()
    else:
        estimator = clf
    
    # fit it to training data
    estimator.fit(X_train, y_train)
    
    # predict using test data
    y_pred = estimator.predict(X_test)
    
    # compute predicted probabilities
    if clf == 'lr':
        return y_pred
    else:
        y_prob = estimator.predict_proba(X_test)
        return y_pred, y_prob


def print_scores(y_test, y_pred, y_prob):
    '''
    Print prediction scores.
    
    Parameters
    ----------
    y_test
        target test data
    y_pred
        target predictions
    y_prob
        target prediction probabilities
    
    Returns
    -------
    prints
        1. Confusion Matrix
        2. Recall Score
            TP / (TP + FN)
        3. Precision Score
            TP / (TP + FP)
        4. F1 Score
            2 * precision * recall / (precision + recall)
        5. Accuracy Score
            (TP + TN) / (TP + FP + TN + FN)
        6. ROC-AUC
            TPR vs FPR curve, scored by area under curve
    graphs
        ROC Curve Graph
    '''
    
    # import libraries
    from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, accuracy_score, f1_score
    from sklearn.metrics import RocCurveDisplay
    
    # confusion matrix
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    
    print('\n--------SCORES---------\n')
    
    # recall scores
    try:
        print('1. Recall Score: ', recall_score(y_test,y_pred))
    except ValueError:
        print('1. Recall Score (multiclass micro)', recall_score(y_test,y_pred, average='micro'))
        print('1. Recall Score (multiclass macro)', recall_score(y_test,y_pred, average='macro'))
    
    # precision scores
    try:
        print('2. Precision Score: ', recall_score(y_test,y_pred))
    except ValueError:
        print('2. Precision Score (multiclass micro)', precision_score(y_test,y_pred, average='micro'))
        print('2. Precision Score (multiclass macro)', precision_score(y_test,y_pred, average='macro'))

    # f1 scores
    try:
        print('3. F1 Score: ', f1_score(y_test,y_pred))
    except ValueError:
        print('3. F1 Score (multiclass micro)', f1_score(y_test,y_pred, average='micro'))
        print('3. F1 Score (multiclass macro)', f1_score(y_test,y_pred, average='macro'))
    
    print('4. Accuracy Score: ', accuracy_score(y_test,y_pred))
    
    # ROC-AUC scores
    try:
        print('5. ROC-AUC: {}'.format(roc_auc_score(y_test, y_prob[:,1])))
    except ValueError:
        print('5. ROC-AUC (multi-class micro ovo): {}'.format(roc_auc_score(y_test, y_prob[:,1], average='micro', multi_class='ovo')))
        print('5. ROC-AUC (multi-class macro ovo): {}'.format(roc_auc_score(y_test, y_prob[:,1], average='macro', multi_class='ovo')))
        print('5. ROC-AUC (multi-class micro ovr): {}'.format(roc_auc_score(y_test, y_prob[:,1], average='micro', multi_class='ovr')))
        print('5. ROC-AUC (multi-class macro ovo): {}'.format(roc_auc_score(y_test, y_prob[:,1], average='macro', multi_class='ovr')))
    except np.AxisError:
        pass
    
    RocCurveDisplay.from_predictions(y_test, y_pred)
    
    
def make_csv(query, filename):
    """ I think it only works for 'SELECT *' statements.
    Will also convert csv file to pandas dataframe. 
    Must call function as a variable.
    (df_name = make_csv(arg1, arg2))
    """
    
    # import libraries
    import psycopg2
    import pandas as pd
    import os
    from pathlib import Path
    
     # check if file already exists
    if os.path.exists(Path('./data') / filename):
        print('File exists. Returning DataFrame...')
        df = pd.read_csv(Path('./data') / filename)
        return df
    
    # ensure all columns are displayed when viewing a pandas dataframe
    pd.set_option('display.max_columns', None)

    # Creating a connection to the database
    print("creating connection...")
    con = psycopg2.connect(database="mid_term_project", 
                           user="lhl_student", 
                           password="lhl_student", 
                           host="lhl-data-bootcamp.crzjul5qln0e.ca-central-1.rds.amazonaws.com", 
                           port="5432")

    # creating a cursor object
    cur = con.cursor()
    # running an sql query
    print("running query...")
    cur.execute(query)
    # Storing the result
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]

    con.close()

    # writing the csv file
    print("writing file...")
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(Path('./data') / filename, index=False)
    
    print("done")