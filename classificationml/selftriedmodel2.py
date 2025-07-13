import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

#  Load the Dataset
file_path = "C:/Users/Dell/Desktop/classificationml/dataset_excavate.xlsx - Sheet 1.csv"
df = pd.read_csv(file_path)
print(df.isnull.sum())