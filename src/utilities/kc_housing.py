## This file contains functions made for developing our linear regression model of 
## King County housing prices
from sqlalchemy import create_engine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import math
from sklearn.feature_selection import RFE 
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings
warnings.filterwarnings(action='once')

def pullsqldata():
    """This function pulls data from three PostGRES tables and returns them into 
    a Pandas Dataframe in order to continue with our EDA. We are filtering records to
    residential housing in 2018. Note: This can take approx. 1 minute to run. """
    engine = create_engine("postgresql:///kc_housing")
    query = """
                SELECT *
                FROM rpsale AS s
                INNER JOIN resbldg AS b ON CONCAT(s.Major,s.Minor) = CONCAT(b.Major, b.Minor)
                INNER JOIN parcel AS p ON CONCAT(s.Major,s.Minor) = CONCAT(p.Major,p.Minor)
                WHERE EXTRACT(YEAR FROM CAST(documentdate AS DATE)) = 2018
                    AND p.proptype = 'R'
                ;"""
    kc_df = pd.read_sql(sql = query, con = engine)
    return kc_df

def clean_data_intial(df):
    """ This function cleans the housing data by removing anomoulous outliers, 
    sale price == 0, and irrelevant columns. It also creates a column, "footprint_ratio"
    based on the size of the house on the lot
    """
    #We chose a minimum sale vale of 100000 and a maximium sale value of 2 sigma
    df_clean = df[(df['saleprice']>100000) & (df['saleprice'] <  (2*df['saleprice'].std())+df['saleprice'].mean())]
    df_clean = df_clean[df_clean['sqftlot'] <  (2*df_clean['sqftlot'].std())+df_clean['sqftlot'].mean()]
    df_clean = df_clean[df_clean['sqfttotliving']<14000]
    #These are irrelevant or highly covariant columns
    columns_to_drop = ['documentdate',
                       'excisetaxnbr',
                       'recordingnbr',
                       'volume',
                       'page',
                       'platnbr',
                       'plattype',
                       'platlot',
                       'platblock',
                       'sellername',
                       'buyername',
                        'streetname',
                        'streettype',
                        'directionsuffix',
                        'buildingnumber',
                        'major',
                        'minor',
                        'bldggradevar',
                        'sqfthalffloor',
                        'sqft2ndfloor',
                        'sqftupperfloor',
                        'sqftunfinfull',
                        'sqftunfinhalf',
                        'sqfttotbasement',
                        'sqftfinbasement',
                        'brickstone',
                        'viewutilization',
                        'propname',
                        'platname',
                        'platlot',
                        'platblock',
                        'range',
                        'township',
                        'section',
                        'quartersection',
                        'area',
                        'subarea',
                        'specarea',
                        'specsubarea',
                        'levycode',
                        'districtname',
                        'currentzoning',
                        'topography',
                        'currentusedesignation',
                        'salewarning',
                        'wetland',
                        'stream',
                        'seismichazard',
                        'landslidehazard',
                        'address',
                        'airportnoise',
                        'contamination',
                        'dnrlease',
                        'coalminehazard',
                        'criticaldrainage',
                        'erosionhazard',
                        'landfillbuffer',
                        'hundredyrfloodplain',
                        'steepslopehazard',
                        'speciesofconcern',
                        'sensitiveareatract',
                        'daylightbasement',
                        'fraction',
                        'directionprefix', 
                        'proptype',
                        'unbuildable', 
                        'bldgnbr', 
                        'pcntcomplete']
    df_clean.drop(columns=columns_to_drop, inplace = True)
    #The columns with Y or N need to be 1 or 0 to model
    df_clean['othernuisances'] = [i.strip() for i in df_clean['othernuisances']]
    df_clean.replace(('Y', 'N'), (1, 0), inplace=True)
    
    #To model the houses that take up more space of thier plot (smaller yard) we need a ratio feature
    #We assume an acturate metric of the house's footprint is the first floor plus any attached garage. This 
    #unfortunatley may not account for detached garages
    df_clean['footprint_ratio']=(df_clean['sqft1stfloor']+df_clean['sqftgarageattached'])/df_clean['sqftlot']
    df_clean.drop(columns = 'sqft1stfloor', inplace = True)
    
    #nbrliving units is classified data telling us if it is a duplex. We want to remove triplexes and create a duplex 
    #flag column. Also the number of triplexes represent a very small portion of our overall dataset
    
    triplex = df_clean.loc[df_clean['nbrlivingunits'] == 3]
    df_clean.drop(triplex.index, inplace= True, axis=0)
    df_clean['duplex'] = df_clean['nbrlivingunits'] - 1
    df_clean.drop(columns = 'nbrlivingunits', inplace = True)
    ratio_drop = df_clean.loc[df_clean['footprint_ratio'] > 1.0]
    df_clean.drop(ratio_drop.index, inplace=True, axis=0)

    
    

    return df_clean    
def recursive_feature_selection(n_features,indep_variables_df, dep_var):
    """
    n_features = number of features to select
    indep_variables = pandas dataframe containing the features to select from
    dep_var = pandas dataframe containing the feature to model \
    returns a list of features to include in model to best fit line
    """
    lr = LinearRegression()
    select = RFE(lr, n_features_to_select=n_features)
    select = select.fit(indep_variables_df, y= dep_var.values.ravel())
    selected_columns = indep_variables_df.columns[select.support_]
    return selected_columns

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

def make_housing_model(list_of_features, df, y):
    """
    Takes in a list of features, a dataframe, and a target (as df['target]). Performs an Ordinary Least Squares (OLS)
    linear regression 
    """
    
    features = df[list_of_features]
    features = sm.add_constant(features)
    model = sm.OLS(y,features).fit()
    
    return model.summary()

def check_feature_linearity(list_of_features, df, y):
    """
    """
    for column in list_of_features:
        plt.scatter(df[column],y, label=column, alpha = .05)
        plt.ylabel('Sale Price')
        plt.xlabel(column)
        plt.title('Linearity Check')
        plt.show()

def check_feature_resid_dist(list_of_features, df, y):
    '''
    Visualizes the residiuals of a linear model in order to check the 
    assumptions. Shows both histogram of residual values and qq plot.
    
    !!!  Be sure to import scipy.stats as stats  !!!
    
    '''
    for feature in list_of_features:
        
        x = df[feature]
        x = sm.add_constant(x)
        model = sm.OLS(y,x).fit()
        pred_val = model.fittedvalues
        residuals = y.values - pred_val
        fig, ax = plt.subplots(1, 2, sharex=False, sharey=False)
        fig.set_size_inches(15,5)
        sns.distplot(residuals, ax = ax[0])
        sm.graphics.qqplot(residuals, dist=stats.norm, fit=True, line='45', ax = ax[1])
        fig.suptitle(feature)
        fig.show()

def check_feature_heteros(list_of_features, df, y):
    """
    Visualizes the heteroscedasticity of a linear model in order to check the 
    assumptions.
    """
    
    for feature in list_of_features:
        x = df[feature]
        x = sm.add_constant(x)
        model = sm.OLS(y,x).fit()
        fig = plt.figure(figsize=(15,8))

        fig = sm.graphics.plot_regress_exog(model, feature, fig=fig)
        plt.show()

def engineer_total_baths(df):
    df['bath_total_count']=df['bathhalfcount']+df['bath3qtrcount']+df['bathfullcount']
    df.drop(columns = ['bathhalfcount','bath3qtrcount','bathfullcount'], inplace = True)
    return df

def engineer_age(df):
    df['age']=2019 - df['yrbuilt']
    df.drop(columns = ['yrbuilt'], inplace = True)
    return df

def engineer_total_porch_space(df):
    df['porch_sqft_total']=df['sqftopenporch']+df['sqftenclosedporch']
    df.drop(columns = ['sqftopenporch','sqftenclosedporch'], inplace = True)
    return df

def zip_code_df(df):
    """
    This function produces a tuple with tuple[0] as a df with the one hot encoded zip code features and tuple[1] as 
    the list of zip code column names. 
    
    The df input should be the dataframe that is output by the "clean_data_initial" function (not a dataframe that 
    the "saleprice" column has been removed from.. this is because we drop rows that do not have a zipcode so we need to 
    keep the shape of the dependent and independent variable dataframes equal). 
    
    """
    #drop the sales that do not include a zip code. We use '98' here to find king county specific zip codes and 
    #we select only the first 5 digits of the zip code because some sales' zip codes have an extraneious 4 digits
    dropped_rows = df[df['zipcode'].str.contains ('98')]
    dropped_rows['zipcode'] = dropped_rows['zipcode'].map(lambda x: x[0:5])

    #use pd.Categorical and pd.get_dummies methods to one hot encode the zip codes
    dropped_rows['zipcode'] = pd.Categorical(dropped_rows['zipcode'])
    df_zip = pd.get_dummies(dropped_rows['zipcode'], prefix = 'zip')
    
    #drop one column from the zip code columns to address the inherent multicoliniearity
    df_zip.drop(columns = 'zip_98000', inplace = True) 
    
    #get a list of zipcode column names to include in model
    list_of_zips = df_zip.columns
    
    #join the zip code dataframe to the dataframe with the other predicitive features
    df_with_zip_cols = dropped_rows.join(df_zip, how = 'inner')
    df_with_zip_cols = df_with_zip_cols.drop(['zipcode'], axis=1)
    
    return df_with_zip_cols, list_of_zips

def make_zipcode_model(df_clean, list_of_baseline_features):
    #call zip_code_df function to produce zip code df and list of zipcodes
    zip_tuple = zip_code_df(df_clean)

    #add on total bath colum using previously used function
    df = engineer_total_baths(zip_tuple[0])
    
    #add on list of other baseline features to the zip code list to put into model
    list_of_features = list(zip_tuple[1])
    list_of_features.extend(list_of_baseline_features)
    
    #produce the model
    
    return make_housing_model(list_of_features, df, df['saleprice'])

def check_zip_code_res_normality(df):
    zip_tuple = zip_code_df(df)
    zip_list = list(zip_tuple[1])
    zip_list.append('saleprice')
    zip_res = zip_tuple[0][zip_list]
    
    lookup_dict = {}
    for col in zip_res.columns:
        
        try:
            index = int(col[-3:])
            search_string = col[-3:]
            amount = int(zip_res[zip_res[col]== True]['saleprice'].mean())
            span = float(zip_res[zip_res[col]== True]['saleprice'].std())
            lookup_dict[col] = (amount, span)
        except:
            continue



    error_list = []       
    for col in zip_res.columns:
        try:
            df_filtered = zip_res[zip_res[col]== True]
            amount = df_filtered['saleprice'].mean()
            span = float(df_filtered['saleprice'].std())
            df_filtered['sigma_difference'] = (df_filtered['saleprice'] - amount)/span
            a = list(df_filtered['sigma_difference'])
            error_list.extend(a)


        except:
            continue  
            
     
    info = list(filter(lambda x: np.abs(x)> 0, error_list))
    
    fig, ax = plt.subplots(1, 2, sharex=False, sharey=False)
    fig.set_size_inches(15,5)
    
    ax[0].set_title('Residuals')
    ax[0].set_xlabel('Standard Deviation')
    ax[0].set_ylabel('Probability')
     
    ax[1].set_title('Heteroscedasticity')
    
    ax[1].set_ylabel('Error')                 
    
    fig.suptitle('Zip Code', fontsize=16)

    
    x = list(range(len(lookup_dict.values())))
    y = [0]*len(x)
    yerr = [x[1] for x in lookup_dict.values()]

    plt.errorbar(x, y, yerr=yerr, fmt='o')
    sns.distplot(info, ax = ax[0])
    return plt.show()

def base_model():
    """calling this function will utilize other defined functions to produce our base model report for King County
    Housing prices - PLEASE NOTE THIS USES A SQL QUERY, MAY TAKE UP TO ONE MINUTE TO COMPLETE
    """
    df_cleaned = clean_data_intial(pullsqldata())
    base_features = ['sqfttotliving','footprint_ratio','duplex']
    Y = df_cleaned['saleprice']
    check_feature_resid_dist(base_features, df_cleaned, Y)
    check_feature_heteros(base_features, df_cleaned, Y)
    check_feature_linearity(base_features, df_cleaned, Y)
    
    return make_housing_model(base_features, df_cleaned, Y)

def waterfront_ohe(final_model_df):
    """this is similar to the zipcode OHE function, but for the waterfront location feature, it drops water_0.0 (not waterfront location)"""
    final_model_df['wfntlocation'] = pd.Categorical(final_model_df['wfntlocation'])
    df_water = pd.get_dummies(final_model_df['wfntlocation'], prefix = 'water')

    #drop the column for houses with no waterfront location
    df_water.drop(columns = 'water_0.0', inplace = True) 
    
    
    #join the zip code dataframe to the dataframe with the other predicitive features
    df_with_water_cols = final_model_df.join(df_water, how = 'inner')
    df_with_water_cols = df_with_water_cols.drop(['wfntlocation'], axis=1)
    return df_with_water_cols

    
