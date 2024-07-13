from credit_fraud_utils_data import load_data, split_data, pipline_preprossing,transform_data,split_target
from credit_fraud_train import Model
from credit_fraud_utils_eval import model_eval,save
import numpy as np

data = load_data('C:\\Users\\omar gamel\\Documents\\New folder\\Credit Card Fraud Detection\\train.csv')

X_train, X_val, y_train, y_val = pipline_preprossing(data)

model = Model(X_train, y_train)

model.train()

rus = model_eval(model.model_train,X_val,y_val)
