import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'bigmart.csv'
bigmart_data = pd.read_csv(file_path)

columns_to_use = [
    'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP',
    'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales'
]
data = bigmart_data[columns_to_use].copy()

data['Item_Weight'] = data['Item_Weight'].fillna(data['Item_Weight'].mean())
data['Outlet_Size'] = data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0])

data['Outlet_Age'] = 2024 - data['Outlet_Establishment_Year']  # Assuming current year is 2024
data['Log_Item_Visibility'] = np.log1p(data['Item_Visibility'])

numerical_cols = ['Item_Weight', 'Item_MRP', 'Outlet_Age', 'Log_Item_Visibility']
categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X = data.drop('Item_Outlet_Sales', axis=1)
y = data['Item_Outlet_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
accuracy = 100 - mape * 100

print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')
print(f'Accuracy: {accuracy}%')


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
sns.regplot(x=y_test, y=y_pred, scatter=False, color='red')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales with Regression Line')
plt.grid(True)
plt.show()

errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, color='purple')
plt.title('Distribution of Prediction Errors with KDE')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Pairplot of Features', y=1.02)
plt.show()

predictions = []
for item_id in bigmart_data['Item_Identifier'].unique():
    for outlet_id in bigmart_data['Outlet_Identifier'].unique():
        sample_data = pd.DataFrame({
            'Item_Weight': [bigmart_data[bigmart_data['Item_Identifier'] == item_id]['Item_Weight'].mean()],
            'Item_Fat_Content': [bigmart_data[bigmart_data['Item_Identifier'] == item_id]['Item_Fat_Content'].mode()[0]],
            'Item_Visibility': [bigmart_data[bigmart_data['Item_Identifier'] == item_id]['Item_Visibility'].mean()],
            'Item_Type': [bigmart_data[bigmart_data['Item_Identifier'] == item_id]['Item_Type'].mode()[0]],
            'Item_MRP': [bigmart_data[bigmart_data['Item_Identifier'] == item_id]['Item_MRP'].mean()],
            'Outlet_Establishment_Year': [bigmart_data[bigmart_data['Outlet_Identifier'] == outlet_id]['Outlet_Establishment_Year'].mode()[0]],
            'Outlet_Size': [bigmart_data[bigmart_data['Outlet_Identifier'] == outlet_id]['Outlet_Size'].mode()[0]],
            'Outlet_Location_Type': [bigmart_data[bigmart_data['Outlet_Identifier'] == outlet_id]['Outlet_Location_Type'].mode()[0]],
            'Outlet_Type': [bigmart_data[bigmart_data['Outlet_Identifier'] == outlet_id]['Outlet_Type'].mode()[0]]
        })
        
        sample_data['Outlet_Age'] = 2024 - sample_data['Outlet_Establishment_Year']
        sample_data['Log_Item_Visibility'] = np.log1p(sample_data['Item_Visibility'])
        sample_data = sample_data.drop(['Item_Visibility', 'Outlet_Establishment_Year'], axis=1)

        sample_data = sample_data[numerical_cols + categorical_cols]

        sample_pred = model.predict(preprocessor.transform(sample_data))[0]

        predictions.append({
            'Item_Identifier': item_id,
            'Outlet_Identifier': outlet_id,
            'Predicted_Sales': sample_pred
        })

future_sales = pd.DataFrame(predictions)
future_sales.to_csv('bigmart_future_sales.csv', index=False)

print(future_sales.head())
