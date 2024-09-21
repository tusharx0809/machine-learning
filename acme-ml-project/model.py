import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pickle

medical_df = pd.read_csv("C:/college/Python/machine-learning/acme-ml-project/medical.csv")

smoker_codes = {
  'no': 0,
  'yes': 1
}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)

sex_codes = {
  'female': 0,
  'male': 1
}
medical_df['sex_code'] = medical_df.sex.map(sex_codes)

#encoding
encoder = OneHotEncoder()
encoder.fit(medical_df[['region']])
one_hot = encoder.transform(medical_df[['region']]).toarray()
medical_df[['northeast','northwest','southeast','southwest']] = one_hot

#scaling
numeric_cols = ['age','bmi','children']
scaler = StandardScaler()
scaler.fit(medical_df[numeric_cols])
scaled_inputs = scaler.transform(medical_df[numeric_cols])

cat_cols = ['smoker_code','sex_code','northeast','northwest','southeast','southwest']
categorical_data = medical_df[cat_cols].values

inputs = np.concatenate((scaled_inputs, categorical_data), axis = 1)
targets = medical_df.charges

model = LinearRegression().fit(inputs, targets)

pickle.dump(scaler, open('C:/college/Python/machine-learning/acme-ml-project/scaler.pkl','wb'))
pickle.dump(encoder, open('C:/college/Python/machine-learning/acme-ml-project/encoder.pkl','wb'))
pickle.dump(model, open('C:/college/Python/machine-learning/acme-ml-project/model.pkl','wb'))