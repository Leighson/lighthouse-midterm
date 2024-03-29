from multiprocessing.sharedctypes import Value
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor

class OutlierExtractor(TransformerMixin):
    from sklearn.pipeline import Pipeline, TransformerMixin
    from sklearn.neighbors import LocalOutlierFactor
    
    def __init__(self, **kwargs):
        """
        Create a transformer to remove outliers. A threshold is set for selection
        criteria, and further arguments are passed to the LocalOutlierFactor class

        Keyword Args:
            neg_conf_val (float): The threshold for excluding samples with a lower
               negative outlier factor.

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        """

        self.threshold = kwargs.pop('neg_conf_val', -10.0)

        self.kwargs = kwargs

    def transform(self, X, y):
        """
        Uses LocalOutlierFactor class to subselect data based on some threshold

        Returns:
            ndarray: subsampled data

        Notes:
            X should be of shape (n_samples, n_features)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        lcf = LocalOutlierFactor(**self.kwargs)
        lcf.fit(X)
        return (X[lcf.negative_outlier_factor_ > self.threshold, :],
                y[lcf.negative_outlier_factor_ > self.threshold])

    def fit(self, *args, **kwargs):
        return self

def split_data(df, target, drop_list=None, scaler=None, test_size=0.2):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

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


def make_csv(query, filename, overwrite=False):
    """
    I think it only works for 'SELECT *' statements.
    Will also convert csv file to pandas dataframe. 
    Must call function as a variable.
    (df_name = make_csv(arg1, arg2))
    
    Query Example
    -------------
    table_name = 'flights_test'
    limit = (10000, )

    query = sql.SQL(
        "SELECT * FROM {table} LIMIT %s").format(
            table = sql.Identifier(table_name),)
    
    filename = 'flights_test_10k_sample.csv'
    """
    
    # import libraries
    import psycopg2
    import pandas as pd
    import os
    from pathlib import Path
    
     # check if file already exists
    if os.path.exists(Path('./data') / filename) and overwrite==False:
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
    
    return(df)


def sql_read_tables(table_name=None):
    '''
    Display database tables as pd.Series object.
    '''
    
    # import libraries
    import psycopg2
    import pandas as pd
    
    # query to return table column names if specified
    if table_name != None:
        query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = table_name;
        """
    else:
        # query to return table names
        query = """
        SELECT tablename
        FROM pg_catalog.pg_tables
        WHERE schemaname != 'pg_catalog'
        AND schemaname != 'information_schema'
        ORDER BY tablename ASC;
        """
            
    # Creating a connection to the database
    con = psycopg2.connect(database="mid_term_project", 
                           user="lhl_student", 
                           password="lhl_student", 
                           host="lhl-data-bootcamp.crzjul5qln0e.ca-central-1.rds.amazonaws.com", 
                           port="5432")

    # creating cursor and run query
    cur = con.cursor()
    cur.execute(query)
    
    # print table names
    tables = [table[0] for table in cur.fetchall()]
    tables = pd.Series(tables)
    
    con.close()
    
    return tables


def unique_values(df):
    '''
    provides all unique values in a presented df
    '''
    for i in range(len(df.columns)):
        unique =  df.iloc[:, i].unique()
        print(i, unique)
    return


def graph_eda(type, x, y=None, bins=20, marker_size=2):
    """
    Quickly graph for EDAs.

    Args:
        type (str): 'hist', 'scatter', 'plot'
        x (list, df.column, array): data for x-axis
        y (list, df.column, array, optional): 'plot' or 'scatter' data for y-axis. Defaults to None.
        bins (int, optional): 'hist' bin sizing to convert continuous data to discrete. HDefaults to 20.
        marker_size (int, optional): 'scatter' marker size. Defaults to 2.

    Returns:
        None: No returns.
    """

    # import libraries
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,8))
    
    # graphs
    if type == 'hist':
        plt.xlabel(x.name)
        plt.ylabel('count')
        plt.hist(x, bins=bins)
        return plt.show()
    
    elif type == 'plot':
        plt.xlabel(x.name)
        plt.ylabel(y.name)
        plt.plot(x,y)
        return plt.show()
    
    elif type == 'scatter':
        plt.xlabel(x.name)
        plt.ylabel(y.name)
        plt.scatter(x,y, s=marker_size)
        return plt.show()


def filter_outliers(df, column):
    """
    Filter outliers in a DataFrame based on z-scores greater than 3. In other words, filter \\
    values that are 3 standard deviations from the column's mean distribution.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be filtered.
    column : pandas.Series
        DataFrame column to be evaluated for outliers.
        
    Returns
    -------
    filter_outliers : pandas>dataFrame
        Return filtered DataFrame.
    """
    
    # import libraries
    from scipy import stats
    
    # remove outliers with |z-score| no greater than 3
    filter_outliers = []
    z = abs(stats.zscore(column))

    for i, score in enumerate(z):
        if score < 3:
            filter_outliers.append(df.iloc[i, :])
            
    df[column] = filter_outliers
    
    return df


def check_normal_dist(data, skipna=True, distribution='norm', bins=20):
    """
    Checks for normal distribution using:
    + Shapiro-Wilk Test
    + Anderson-Darling Test
    + Skewness and Kurtosis
    
    Plots distribution according to number of bins passed.
    
    Parameters
    ---------
    data : array, list, pandas.Series
        Distribution to check normality.
    skipna : bool | default=True
        Ignores null or na values in data to determine 'kurtosis' and 'skewness'.
    distribution : str | default='norm'
        Accepts: 'norm', 'expon', 'logistic', 'gumbel', 'gumbel_l', 'gumbel_r', or 'extreme1'
        Check `stats.anderson()` for further documentation.
    bins : int | default=20
        Defines number of bins for plot.
    
    Returns
    -------
    None
    """
    # import libraries
    from scipy import stats
    import matplotlib.pyplot as plt
    
    # shapiro test
    print('Shapiro-Wilk Test')
    print('-----------------')
    print('Provided an alpha of 0.05, if p-value > alpha then the distribution can be assumed to be normal.')
    print('This test prunes data to the first 5000 data points or "head(5000)" as reliability suffers with increasing samples.')
    s, p = stats.shapiro(data.head(5000))
    print('t-stat:', s, ', p-value:', p)

    # anderson test
    print('\nAnderson-Darling Test')
    print('---------------------')
    print('If the returned statistic > critical values then for the corresponding significance level,')
    print('the null hypothesis that the data come from the chosen distribution can be rejected.')
    stat, crit, sig = stats.anderson(data, distribution)
    print('t-stat:', stat, ', critical values:', crit, ', at significance levels of:', sig)

    # skewness and kurtosis
    print('\nSkewness and Kurtosis')
    print('---------------------')
    print('skewness, -ve skews left and +ve skews right (0 is best):', data.skew(skipna=skipna))
    print('kurtosis, tail spread (< 3 is best):', data.kurtosis(skipna=skipna))
    
    # plot distribution
    graph_eda('hist', x=data, bins=bins)


def search_data(data, regex_term):
    """
    Use regex formatted terms to search through any list-like data. Returns results in list form.
    This will also return indices in case DataFrame row filtering is required.
    
    For RegularExpression syntax: https://www.programiz.com/python-programming/regex
    
    Note: Remember to reset DataFrame indexing if previously filtered. The resulting
    indices here only 'remembers' positions--it will not account for non-sequential
    indexes.
    
    Parameters
    ----------
    data : str of list, array, pandas.Series
        Data to loop search through. Must be strings.
    regex_term : regex str
        Format as 'r"<search terms>"'. Exclude arrow/tag brackets.
    
    Returns
    -------
    indices : list
        Positional indices.
    results : list
        Resulting string objects matching regex search.
    """
    # import libraries
    import re
    
    regex = regex_term

    # search 
    results = []
    indices = []
    for i, values in enumerate(data):
        result = re.search(regex, values)
        
        if result != None:
            results.append(result.group(0))
            indices.append(i)

    return indices, results


def sql_read(query):
        # import libraries
        import pandas as pd
        import psycopg2 as pg
    
        con = pg.connect(database="mid_term_project", 
                                user="lhl_student", 
                                password="lhl_student", 
                                host="lhl-data-bootcamp.crzjul5qln0e.ca-central-1.rds.amazonaws.com", 
                                port="5432")

        # create cursor object
        cur = con.cursor()
        
        # run sql query
        cur.execute(query)
        
        # store result to rows and cols
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]

        # close connection
        con.close()

        # convert to df and return
        return pd.DataFrame(rows, columns=cols)


def sql_search_date(table, field='fl_date', y=2019, m=None, d=None, limit=1000, overwrite=False):
    """Search SQL database based on dates (yyyy-mm-dd).
    Returns matching records (rows).
    
    - year-month search: If only searching for specific months, increase the default limit up to 25k. Monthly record counts don't go much further than 25k.
    
    - year or year-day search: Otherwise, searching the entire year or specific days of the year (regardless of month) is computationally expensive so
    it is recommended that sample pulls be kept at default. Mechanically, this function will randomly sample 1K records per month, sequentially appending them
    to a single DataFrame before returning the result. If a day is specified, it will also filter by day afterwards.
    
    - year-month-day search: The simplest search is for a specific date.  Feel free to increase the default limit. It's probably not an issue pulling complete
    records of that date.
    
    Post search, it will also attempt to write the df to csv granted that the file does not already exist locally.
    It will return the dataframe results regardless

    Args:
        table (str): SQL table to seardh.
        field (srt): SQL field (column) to search.
        y (int, optional): Year. Defaults to 2019.
        m (int, optional): Month. Defaults to None.
        d (int, optional): Day. Defaults to None.
        term (dict, optional): {<feature/field> : <search term>}, key is the field to search, value is the search term
        limit (int, optional): n limit to records returned. Defaults to 1000.
        overwrite (bool, optional): If the file exists, write to csv will not proceed.

    Returns:
        DataFrame: Records returned as DataFrame.
    """
    
    # import libraries
    from psycopg2 import sql
    from pathlib import Path
    import pandas as pd
    import os
    
    # define common ql variables
    fields = sql.Identifier(field)
    tables = sql.Identifier(table)
    years = sql.Literal(y)
    limits = sql.Literal(limit)
    sample_size = int(limit/1000)
    
    if field == 'fl_date' and table == 'flights':
    
        if (m != None) and (d != None): # filter for month and day, if specified
            
            # check if file already exists
            # if it sees local file, it only returns df
            filename = '{}_{}K_y{}m{:02d}d{:02d}_sample.csv'.format(table, sample_size, y, m, d)
            if os.path.exists(Path('./data') / filename) and overwrite==False:
                print('File exists. Returning DataFrame...')
                df = pd.read_csv(Path('./data') / filename)
                return df
            
            # define search-specific sql variables
            months = sql.Literal(m)
            days = sql.Literal(d)
            
            if m < 10: # for months/days less than 10; add leading 0
                if d < 10:
                    query = sql.Composed([sql.SQL("SELECT * FROM {tbl} ").format(tbl=tables),
                                        sql.SQL("WHERE {fld} ~* '^{yr}-0{mon}-0{dy}' ").format(fld=fields, yr=years, mon=months, dy=days),
                                        sql.SQL("ORDER BY RANDOM() LIMIT {lim};").format(lim=limits)])
                else:
                    query = sql.Composed([sql.SQL("SELECT * FROM {tbl} ").format(tbl=tables),
                                        sql.SQL("WHERE {fld} ~* '^{yr}-0{mon}-{dy}' ").format(fld=fields, yr=years, mon=months, dy=days),
                                        sql.SQL("ORDER BY RANDOM() LIMIT {lim};").format(lim=limits)])
            else:
                if d < 10:
                    query = sql.Composed([sql.SQL("SELECT * FROM {tbl} ").format(tbl=tables),
                                        sql.SQL("WHERE {fld} ~* '^{yr}-{mon}-0{dy}' ").format(fld=fields, yr=years, mon=months, dy=days),
                                        sql.SQL("ORDER BY RANDOM() LIMIT {lim};").format(lim=limits)])
                else:
                    query = sql.Composed([sql.SQL("SELECT * FROM {tbl} ").format(tbl=tables),
                                        sql.SQL("WHERE {fld} ~* '^{yr}-{mon}-{dy}' ").format(fld=fields, yr=years, mon=months, dy=days),
                                        sql.SQL("ORDER BY RANDOM() LIMIT {lim};").format(lim=limits)])
            
            # save to df and name file output
            df = (sql_read(query))
        
        elif (m != None): # filter for month only, if specified
            
            # check if file already exists
            # if it sees local file, it only returns df
            filename = '{}_{}K_y{}m{:02d}d00_sample.csv'.format(table, sample_size, y, m)
            if os.path.exists(Path('./data') / filename) and overwrite==False:
                print('File exists. Returning DataFrame...')
                df = pd.read_csv(Path('./data') / filename)
                return df
            
            # define search-specific sql variables
            months = sql.Literal(m)
            
            if m < 10: # for month less than 10; add leading 0
                query = sql.Composed([sql.SQL("SELECT * FROM {tbl} ").format(tbl=tables),
                                    sql.SQL("WHERE {fld} ~* '^{yr}-0{mon}' ").format(fld=fields, yr=years, mon=months),
                                    sql.SQL("ORDER BY RANDOM() LIMIT {lim};").format(lim=limits)])
            else:
                query = sql.Composed([sql.SQL("SELECT * FROM {tbl} ").format(tbl=tables),
                                    sql.SQL("WHERE {fld} ~* '^{yr}-{mon}' ").format(fld=fields, yr=years, mon=months),
                                    sql.SQL("ORDER BY RANDOM() LIMIT {lim};").format(lim=limits)])
            
            # save to df and name file output
            df = (sql_read(query))
            
        elif (d != None): # if specified, filter for specific days of the year (month-agnostic)
            
            # check if file already exists
            # if it sees local file, it only returns df
            filename = '{}_{}K_y{}m00d{:02d}_sample.csv'.format(table, sample_size, y, d)
            if os.path.exists(Path('./data') / filename) and overwrite==False:
                print('File exists. Returning DataFrame...')
                df = pd.read_csv(Path('./data') / filename)
                return df
            
            # define search-specific sql variables
            days = sql.Literal(d)
            
            if d < 10: # for month less than 10; add leading 0
                query = sql.Composed([sql.SQL("SELECT * FROM {tbl} ").format(tbl=tables),
                                    sql.SQL("WHERE {fld} ~* '^{yr}-[0-9][0-9]-0{dy}$' ").format(fld=fields, yr=years, dy=days),
                                    sql.SQL("ORDER BY RANDOM() LIMIT {lim};").format(lim=limits)])
            else:
                query = sql.Composed([sql.SQL("SELECT * FROM {tbl} ").format(tbl=tables),
                                    sql.SQL("WHERE {fld} ~* '^{yr}-[0-9][0-9]-{dy}$' ").format(fld=fields, yr=years, dy=days),
                                    sql.SQL("ORDER BY RANDOM() LIMIT {lim};").format(lim=limits)])
            
            # save to df and name file output
            df = (sql_read(query))
        
        else: # no month defined; returns entire year by default, monthly samples constrained to limit
            
            # check if file already exists
            # if it sees local file, it only returns df
            filename = '{}_{}K_y{}m00d00_sample.csv'.format(table, sample_size, y)
            if os.path.exists(Path('./data') / filename) and overwrite==False:
                print('File exists. Returning DataFrame...')
                df = pd.read_csv(Path('./data') / filename)
                return df
            
            # force january lookup to instantiate df
            query = sql.Composed([sql.SQL("SELECT * FROM {tbl} ").format(tbl=tables),
                                sql.SQL("WHERE {fld} ~* '^{yr}-01' ").format(fld=fields, yr=years),
                                sql.SQL("ORDER BY RANDOM() LIMIT {lim};").format(lim=limits)])
        
            # define dataframe to return
            df = (sql_read(query))
            
            # loop through all months to sample, then concatenate to df
            for month in range(2,13):
                
                # define search-specific sql variables
                months = sql.Literal(month)
                
                if month < 10: # for month less than 10; add leading 0
                    query = sql.Composed([sql.SQL("SELECT * FROM {tbl} ").format(tbl=tables),
                                        sql.SQL("WHERE {fld} ~* '^{yr}-0{mon}' ").format(fld=fields, yr=years, mon=months),
                                        sql.SQL("ORDER BY RANDOM() LIMIT {lim};").format(lim=limits)])

                    # append to existing df
                    df = pd.concat([df, sql_read(query)])
                
                else:
                    # for months more than 9; month has no leading 0
                    query = sql.Composed([sql.SQL("SELECT * FROM {tbl} ").format(tbl=tables),
                                        sql.SQL("WHERE {fld} ~* '^{yr}-{mon}' ").format(fld=fields, yr=years, mon=months),
                                        sql.SQL("ORDER BY RANDOM() LIMIT {lim};").format(lim=limits)])
                    
                    # append to existing df
                    df = pd.concat([df, sql_read(query)])
                    
    elif table == 'passengers':
        
        filename = '{}_{}K_y{}_sample.csv'.format(table, sample_size, y)
        
        if os.path.exists(Path('./data') / filename) and overwrite==False:
            print('File exists. Returning DataFrame...')
            df = pd.read_csv(Path('./data') / filename)
            return df
        
        months = [i for i in range(1,13)]
        for m in months:
            months = sql.Literal(m)
            query = sql.Composed(
                [sql.SQL("SELECT * FROM {tbl} ").format(tbl=tables),
                sql.SQL("WHERE {fld} ~* '^{mo}' ").format(fld=fields, mo=months),
                sql.SQL("ORDER BY RANDOM() LIMIT {lim};").format(lim=limits)]
                )

        # save to df and name file output
        df = (sql_read(query))
        
    elif table == '':
        
        filename = '{}_{}K_y{}_sample.csv'.format(table, sample_size, y)
    
    # writes csv file and returns df
    print("Writing file...")
    df.to_csv(Path('./data') / filename, index=False)
    print("Returning DataFrame...")
    return df

  
def replace_with_numeric(df, column):
    '''
    input the data frame and the column to replace the unique values with numeric values
    '''
    unique_vals = df[column].unique()
    df[column].replace(to_replace=unique_vals,
                                  value= list(range(len(unique_vals))),
                                  inplace=True)
    return


def xgboost_det(df, target, params, cv_params, features, drop=None, weight=None, scaler=None, gridsearch=False):
    """
    Set the params for XGBoost and cross-validation using XGBoost method.
    Optionally set to optimize hyperparameters using GridSearchCV before performing
    the cross validation.
    
    https://xgboost.readthedocs.io/en/latest/parameter.html
    
    Hyperparameters:
    
    PARAMS
        'objective' : 'reg:squarederror', 'reg:logistic', 'binary:logistic', or 'multi:softmax'
         - 'num_class' : int, set this according to class count if using 'multi:softmax' as objective.
        'colsample_bytree' : 1, subsample ratio of columns when constructing each tree.
        'learning_rate' : 0.3, step size shrinkage used in update to prevents overfitting. 
        'max_depth' : 6 maximum depth of a tree. Decrease to diminish overfitting. Computationally expensive.
        'lambda' : 1, L2 regularization. Increase to make model more conservative.
        'alpha' : 0, L1 regularization. Increase to make model more conservative.
        'n_estimators' : set number of 
        trees for the ensemble.
        
    CV_PARAMS
        'nfold' : 5, number of k-folds.
        'num_boost_round' : int, same as 'n_estimators'. Can set different number here.
        'early_stopping_rounds' : 10, stop count for validation set if the performance doesn't improve.
        
    Args:
        df (pd.DataFrame): _description_
        target (str): pass the column/field name of target variable as string.
        params (dict): params.keys as parameter names, params.values as parameter values.
        scaler (str, optional): 'standard' or 'minmax'. Defaults to None.
        gridsearch (bool, optional): performs GridSearchCV to optimize for hyperparameters. Defaults to False.
    """
    
    # import libraries
    import xgboost as xgb
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV
    
    # split data into training and scale
    # decision tree based algorithms are not sensitive to scaling, but it wouldn't hurt
    # in our case, min/max scaling would be appropriate because our target is not normally distributed
    X_train, X_test, y_train, y_test = split_data(df, target, drop_list=drop, scaler=scaler)
    
    # create training data structured using XGBoost DMatrix
    # for use during cross validation
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    
    # instantiate XGBoost regression model
    xg_reg = xgb.XGBRegressor()
    
    # fit data to model and predict using test data from split_data
    xg_reg.fit(X_train, y_train)
    y_pred = xg_reg.predict(X_test)
    
    xg_reg.get_booster().feature_names = features
    
    # calculate initial RMSE score using predicted and test targets
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE: %f' % (rmse))
    
    #plots
    plt.rcParams['figure.figsize'] = [10, 8]
    
    # plot decision tree
    # xgb.plot_tree(xg_reg, num_trees=0)
    # plt.show()
    
    # plot feature importance
    xgb.plot_importance(xg_reg.get_booster())
    plt.show()
    
    chart = plt.gcf()
    
    chart.savefig('chart.png', dpi = 300)

    if gridsearch == True:
        clf = GridSearchCV(estimator=xg_reg, param_grid=params, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # view the accuracy score and best hyperparameter settings
        print('Best score for training:', clf.best_score_)
        print('Best hyperparameters: ', clf.best_params_)
        
        # apply classifier to testing dataset using best params
        clf.score(X_test, y_test)
    
    # with as_pandas set to True, results in df of cv_results
    cv_results = xgb.cv(
        data=X_train,
        params=clf.best_params_,
        nfold=cv_params['nfold'],
        num_boost_round=cv_params['num_boost_round'],
        early_stopping_rounds=cv_params['early_stopping_rounds'],
        metrics="rmse",
        as_pandas=True)
    
    # cv_results are separated into train and test datasets.
    # each describes scores based on the set metrics to evaluate.
    # in this case, it will return 'rsme' scores as mean and std
    # aggregations of the folds.
    
    print('CV Results:', cv_results)
    return y_pred, cv_results


def xgboost(X_train, X_test, y_train, y_test, n_estimators = 10, max_depth = 5, alpha = 10, num_boost_round = 50):
    '''
    Set the params for XGBoost and perform cross validation by K-folds method, will eventually split the functions for more specificity
    '''
     # import libraries
    import xgboost as xgb
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV
    
    xg_reg = xgb.XGBRegressor(objective = 'reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, 
                          max_depth = max_depth, alpha = alpha, n_estimators = n_estimators)
    xg_reg.fit(X_train, y_train)
    y_pred = xg_reg.predict(X_test)
    
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    
    # calculate initial RMSE score using predicted and test targets
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE: %f' % (rmse))
        
    # with as_pandas set to True, results in df of cv_results
    cv_results = xgb.cv(
        dtrain=dtrain,
        params= dict(
            objective = 'reg:linear',
            colsample_bytree = 0.3,
            learning_rate = 0.1,
            max_depth = max_depth,
            alpha = alpha,
            n_estimators = n_estimators),
        nfold=5,
        num_boost_round=max_depth,
        early_stopping_rounds=10,
        metrics="rmse",
        as_pandas=True
        )
    
    print(cv_results)
    return


def date_to_dtday(df, feature, form = 'day'):
    '''
    set date column to either binary for grouping of weekd and weekend
    '''

    df[feature] =  pd.to_datetime(df[feature], format='%Y-%m-%d')
    df[feature] = df[feature].dt.day_of_week
    if form == 'binary':
        df.loc[df[feature] <= 4, feature] = 0
        df.loc[df[feature] > 4, feature] = 1
    return df


def set_class_grouping(df, feature):
    '''
    set class values for class feature - highly specific, only to be used for iterations of passenger table
    '''
    df.loc[df[feature] == 'A', feature] = 0
    df.loc[df[feature] == 'F', feature] = 0
    df.loc[df[feature] == 'C', feature] = 1
    df.loc[df[feature] == 'J', feature] = 1
    df.loc[df[feature] == 'R', feature] = 1
    df.loc[df[feature] == 'D', feature] = 1
    df.loc[df[feature] == 'I', feature] = 1
    df.loc[df[feature] == 'W', feature] = 2
    df.loc[df[feature] == 'P', feature] = 2
    df.loc[df[feature] == 'Y', feature] = 3
    df.loc[df[feature] == 'K', feature] = 3
    df.loc[df[feature] == 'M', feature] = 3
    df.loc[df[feature] == 'L', feature] = 3
    df.loc[df[feature] == 'G', feature] = 3
    df.loc[df[feature] == 'V', feature] = 3
    df.loc[df[feature] == 'S', feature] = 3
    df.loc[df[feature] == 'N', feature] = 4
    df.loc[df[feature] == 'Q', feature] = 4
    df.loc[df[feature] == 'O', feature] = 4
    df.loc[df[feature] == 'E', feature] = 4
    df.loc[df[feature] == 'B', feature] = 5
    return df.head()
