import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
cust = pd.read_csv('customer_churn.csv')
x = cust.drop(columns=['Names', 'Onboard_date', 'Location', 'Company', 'Churn'], axis=1)
y = cust['Churn']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Save model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
