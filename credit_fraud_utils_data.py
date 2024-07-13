import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter



def load_data(file_path):
    '''
    this function to loading data
    file_path: URL of data.
    return: data without any processing.
    '''
    df=pd.read_csv(file_path)
    return df

def split_target(data, target='Class'):
    X=data.drop(target,axis=1)
    y=data[target]
    return X, y


def split_data(full_data,y, test_size=.2,random_state=11, shuffle=False):
    '''

    '''
   
    X_train, X_test, y_train, y_test=train_test_split(full_data,y,test_size=test_size,random_state=random_state,shuffle=shuffle)
    return X_train, X_test, y_train, y_test

# missing values , cheak imbalanced dataset, handel impalanced, outlier

def under_sample(x, y, factor):
    counter = Counter(y)    
    minority_size = counter[1]  
    new_sz = int(minority_size * factor)
    #ratio of the number of samples in the majority(neg) class to the minority(pos) class
    rus = RandomUnderSampler(random_state=11, sampling_strategy={0:new_sz})
    X_resampled, y_resampled = rus.fit_resample(x, y)

    return X_resampled, y_resampled


def check_imbalanced_data(df,target='Class', threshold=.1):
    # Calculate class distribution
    class_distribution = df[target].value_counts(normalize=True)
    print(class_distribution)
    # Check if the ratio of minority class to majority class is less than the threshold
    print(class_distribution.min())
    print(class_distribution.max())
    minority_class_ratio = class_distribution.min() / class_distribution.max()
    
    return minority_class_ratio < threshold


def handel_imbalanced_data( df, target='Class', threshold=.1,factor=1 ):
    X, y=split_target(df)

    X_resampled, y_resampled=under_sample(X,y,factor)

    sup_sample=pd.concat([X_resampled,y_resampled],axis=1)
    sup_sample=sup_sample.sample(frac=1)
    return sup_sample
    


def check_missing_values(data):
    return data.isna().sum().max() > 0


def handle_missing_values(data):
    
    if check_missing_values(data):
        # Separate numerical and categorical columns
        numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()

        # Handling missing values for numerical columns
        for col in numerical_cols:
            if data[col].isnull().any():
                # Fill missing values in numerical columns with the mean
                data[col].fillna(data[col].mean(), inplace=True)

        # Handling missing values for categorical columns
        for col in categorical_cols:
            if data[col].isnull().any():
                # Fill missing values in categorical columns with the mode
                data[col].fillna(data[col].mode()[0], inplace=True)

        return data
    
    return data



def handle_outliers(data):
    # Handling outliers in numerical columns
    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    for col in numerical_cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data = data.drop(data[(data[col] > upper_bound) | (data[col] < lower_bound)].index)

    # Handling outliers in categorical columns (encode rare categories)
    categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    for col in categorical_cols:
        counts = data[col].value_counts(normalize=True)
        rare_categories = counts[counts < 0.05].index  # Consider categories with less than 5% occurrence as rare
        data[col] = np.where(data[col].isin(rare_categories), 'Other', data[col])

    return data

def transform_data(data):
    #split at first and scaling of training data then val data by min max train data to avoid data leakage 
    processor =RobustScaler()
    data_scaling=processor.fit_transform(data)

    return processor, data_scaling


def pipline_preprossing(data):
    data = handle_missing_values(data)
    #data = handle_outliers(data)
    new_data = handel_imbalanced_data(data)
    x,y= split_target(new_data)
    X_train, X_test, y_train, y_test = split_data(x,y)

    rob_scaler = RobustScaler()

    X_train['scaled_amount'] = rob_scaler.fit_transform(X_train['Amount'].values.reshape(-1,1))
    X_train['scaled_time'] = rob_scaler.fit_transform(X_train['Time'].values.reshape(-1,1))
    
    X_test['scaled_amount'] = rob_scaler.fit_transform(X_test['Amount'].values.reshape(-1,1))
    X_test['scaled_time'] = rob_scaler.fit_transform(X_test['Time'].values.reshape(-1,1))
    X_train.drop(['Time','Amount'], axis=1, inplace=True)
    X_test.drop(['Time','Amount'], axis=1, inplace=True)

    return X_train, X_test, y_train, y_test 






    



