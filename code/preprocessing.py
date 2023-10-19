
import prep_helper

import argparse
import os
import glob
import warnings
import tarfile

import numpy as np
import pandas as pd
import datetime as datetime

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.exceptions import DataConversionWarning
import prep_helper

warnings.filterwarnings(action="ignore", category=DataConversionWarning)   

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--test-size", type=float, default=0.2)    
    parser.add_argument("--input-path", type=str, default='/opt/ml/processing/input')
    parser.add_argument("--label", type=str, default='Churn')    
    args, _ = parser.parse_known_args()
    
    bucket = 'markos-telco-churn'
    csv_file_path = 'ingest/Telco-Customer-Churn.csv'

    # Define the S3 URL for the CSV file
    s3_url = f's3://{bucket}/{csv_file_path}'

    # Use pandas to read the CSV file
    df = pd.read_csv(s3_url)
    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
    df.SeniorCitizen = df.SeniorCitizen.apply(str)

    #Removing missing values 
    df.dropna(inplace = True)
    
    # imporove column values
    df = prep_helper.improve_column_val(df)
    
    #Remove customer IDs from the data set
    df2 = df.iloc[:,1:]
    #Convertin the predictor variable in a binary numeric variable
    df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
    df2['Churn'].replace(to_replace='No',  value=0, inplace=True)   

    # Split train test val set
    label = args.label
    train_size = args.train_size
    test_size = args.test_size

    print("Splitting data into {:.0%} train, {:.0%} test, and {:.0%} validation sets".format(train_size,test_size,(1-train_size-test_size)))
    X_train, X_test, y_train, y_test = train_test_split(
        df2.drop(label, axis=1), df2[label], train_size=train_size, random_state=0)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=(1-test_size), random_state=0)  


    preprocess = make_column_transformer(
        (StandardScaler(),
        make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(drop='first',dtype='int'),
        make_column_selector(dtype_include=object)))

    col_names = ['Churn', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_M', 'SeniorCitizen_Y', 
                 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 
                 'MultipleLines_No_phone', 'MultipleLines_Yes', 
                 'InternetService_Fiber', 'InternetService_No', 'OnlineSecurity_No_internet', 
                 'OnlineSecurity_Yes','OnlineBackup_No_internet', 'OnlineBackup_Yes',
                 'DeviceProtection_No_internet', 'DeviceProtection_Yes', 'TechSupport_No_internet', 
                 'TechSupport_Yes', 'StreamingTV_No_internet', 'StreamingTV_Yes',
                 'StreamingMovies_No_internet', 'StreamingMovies_Yes', 'Contract_One_year',
                 'Contract_Two_years', 'PaperlessBilling_Yes',
                 'PaymentMethod_Credit_card', 'PaymentMethod_Electronic_check', 'PaymentMethod_Mailed_check']


    # Fit preprocessor on train data and transform both test and train dataset
    train_features = preprocess.fit_transform(X_train)
    test_features = preprocess.transform(X_test)
    val_features = preprocess.transform(X_val)    

    #Insert label column as first column
    train_features = np.insert(train_features,0,y_train, axis=1)
    test_features = np.insert(test_features,0,y_test,axis=1)
    val_features = np.insert(val_features,0,y_val,axis=1)

    train = pd.DataFrame(train_features, columns=col_names)
    test = pd.DataFrame(test_features, columns=col_names)
    val = pd.DataFrame(val_features, columns=col_names)    

    print("Train data shape after preprocessing: {}".format(train_features.shape))
    print("Test data shape after preprocessing: {}".format(test_features.shape))
    print("Validation data shape after preprocessing: {}".format(val_features.shape))    

    train_output_path = os.path.join("/opt/ml/processing/train", "train.csv")
    test_output_path = os.path.join("/opt/ml/processing/test", "test.csv")
    val_output_path = os.path.join("/opt/ml/processing/val", "val.csv")    
    model_output_path = "/opt/ml/processing/model"
    model_filename = "model.tar.gz"

    print("Saving training data frame to {}".format(train_output_path))
    train.to_csv(train_output_path, header=True, index=False)
    print("Saving test data frame to {}".format(test_output_path))
    test.to_csv(test_output_path, header=True, index=False)
    print("Saving val data frame to {}".format(val_output_path))
    val.to_csv(val_output_path, header=True, index=False)    

    print("Job was successful!")
