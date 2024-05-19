###### OLS(Bakward Selection) and GB - y=revenue, only superhost model ###########

import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


cities = ["Chicago", "Dallas", "Houston", "Oakland", "Washington", "Boston", 'Los Angeles', 'Miami', 'Philadelphia']
# cities = ["Chicago", 'Los Angeles']
# Assuming your dataset is stored in a CSV file named 'your_dataset.csv'
df = pd.read_csv("airbnb_New York.csv")

for city in cities:
    df1 = pd.read_csv(f"airbnb_{city}.csv")
    df = pd.concat([df, df1], ignore_index=True)

# Convert the 'Scraped Date' column to datetime format
df['Scraped Date'] = pd.to_datetime(df['Scraped Date'])
df = df[(df['Scraped Date'] >= '2020-05-01') & (df['Scraped Date'] <= '2020-06-30')]

# Drop varaibles not in list
df = df.drop(['prev_time_to_date_mean','prev_scrapes_in_period','superhost_observed_in_period',
              'Integrated Property Manager', 'City_y', 'scrapes_in_period', 'superhost_date_diff',
              'superhost_change', 'hostResponseNumber_pastYear', 'hostResponseAverage_pastYear',
              'prev_hostResponseNumber_pastYear', 'prev_hostResponseAverage_pastYear',
              'time_to_date_mean', 'Property Type.1'], axis=1)

# Drop varaibles related to the superhost quaterias
df = df.drop(['superhost_change_gain_superhost', 'superhost_change_lose_superhost','prev_host_is_superhost','prev_host_is_superhost_in_period','superhost_period_all','prev_superhost_period_all','rating_ave_pastYear', 'numReviews_pastYear', 'numCancel_pastYear', 'num_5_star_Rev_pastYear',
             'prop_5_StarReviews_pastYear', 'prev_rating_ave_pastYear', 'prev_numReviews_pastYear', 'prev_numCancel_pastYear',
             'prev_num_5_star_Rev_pastYear', 'prev_prop_5_StarReviews_pastYear', 'Number of Reviews', 'prev_Number of Reviews',
             'Rating Overall', 'prev_Rating Overall', 'prev_revenue', 'prev_host_is_superhost1', 'prev_year_superhosts'], axis = 1)


# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['City_x', 'Listing Type', 'Pets Allowed'])
# df = pd.get_dummies(df, columns=['Listing Type', 'Pets Allowed'])



#### dfs - for super host #######
# Drop 'Integrated Property Manager' from the list of columns
dfs = df.drop(['Airbnb Host ID', 'Airbnb Property ID', 'Zipcode', 'Latitude', 'Longitude', 'Neighborhood', 'Property Type'], axis=1)

dfs = dfs[(dfs['host_is_superhost_in_period']== 1)]
# Drop superhost as independent variables
dfs = dfs.drop(['host_is_superhost_in_period', 'Superhost'], axis=1)
dfs = dfs.dropna()

# Define independent variables (X) and the target variable (y)
X = dfs.drop(['revenue', 'Created Date', 'Scraped Date'], axis=1)  # Drop irrelevant columns
y = np.log(dfs['revenue'])
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=42)

# print(X.head())
# Add a constant term to the independent variables

# Perform backward selection
def backward_selection(X, y, threshold=0.05):
    while True:
        model = sm.OLS(y, X).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()
        if max_p_value > threshold:
            # Remove the predictor with the highest p-value
            remove_variable = p_values.idxmax()
            X = X.drop(remove_variable, axis=1)
        else:
            break
    return model

final_model = backward_selection(X_train, y_train)

# Display final model summary
print(final_model.summary())

# print the selected variables

final_model_vars = final_model.params.index.tolist() #   [1:]Excluding the constant term

# print("Variables in the Final Model:")
# print(final_model_vars)


### MSE for Regression####

from sklearn.model_selection import train_test_split
import numpy as np


print('')
print('MSE for Regression')
# Define the selected features
selected_features = final_model_vars


# Get the predictions for the training set
y_train_pred = final_model.predict(sm.add_constant(X_train[selected_features]))

# Calculate the MSE for the training set
mse_train = np.mean((y_train - y_train_pred) ** 2)
print(f"Mean Squared Error (MSE) for the Training Set: {mse_train:.10f}")

# Get the predictions for the testing set
y_validation_pred = final_model.predict(sm.add_constant(X_validation[selected_features]))
# y_validation_pred = final_model.predict(X_validation[selected_features])


# Calculate the MSE for the testing set
mse_validation = np.mean((y_validation - y_validation_pred) ** 2)
print(f"Mean Squared Error (MSE) for the Testing Set: {mse_validation:.10f}")


####### GB & MSE of GB ######
print('')
print('MSE for GB')


# Initialize Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit the model on the training data
gb_final_model = gb_regressor.fit(X_train, y_train)
selected_features_gb = X_train.columns[gb_regressor.feature_importances_ != 0]

# Print details about the Gradient Boosting model
print("Gradient Boosting Regressor Details:")
print(f"Number of Estimators: {gb_regressor.n_estimators}")
print(f"Learning Rate: {gb_regressor.learning_rate}")
print(f"Maximum Depth: {gb_regressor.max_depth}")
print(f'The selected variables are {selected_features_gb}')

# Predict on the validation set and
y_validation_pred = gb_regressor.predict(X_validation)
y_train_pred = gb_regressor.predict(X_train)


# Calculate the MSE for the validation set
mse_validation = mean_squared_error(y_validation, y_validation_pred)
mse_train = mean_squared_error(y_train, y_train_pred)

print(f"Mean Squared Error (MSE) for the Train Set: {mse_train:.10f}")
print(f"Mean Squared Error (MSE) for the Validation Set: {mse_validation:.10f}")


from sklearn.metrics import r2_score

# Assuming y_validation and y_validation_pred are your actual and predicted values
r_squared = r2_score(y_validation, y_validation_pred)
print(f"R-squared: {r_squared:.4f}")


############### PART 2 - use GB to run revenue for non-superhost ##########################

#### dfy- for non superhost#####
dfy = df.drop(['Zipcode', 'Latitude', 'Longitude', 'Neighborhood', 'Property Type'], axis=1)
dfy = dfy[(dfy['host_is_superhost_in_period'] == 0)]

# Drop superhost as independent variables
dfy = dfy.drop(['host_is_superhost_in_period', 'Superhost'], axis=1)
dfy = dfy.dropna()

# Define independent variables (X) and the target variable (y)
X_ns = dfy.drop(['Airbnb Host ID', 'Airbnb Property ID','revenue', 'Created Date', 'Scraped Date'], axis=1)  # Drop irrelevant columns
y_ns = np.log(dfy['revenue'])


# nonsuperhost_revenue_predict = gb_final_model.predict(X_ns)

nonsuperhost_revenue_predict = gb_final_model.predict(sm.add_constant(X_ns))


results_df = pd.DataFrame({
    'Airbnb Host ID': dfy['Airbnb Host ID'],
    'Airbnb Property ID': dfy['Airbnb Property ID'],
    'Actual Revenue': dfy['revenue'],
    'Predicted Revenue for Non-Superhost (natural log)': nonsuperhost_revenue_predict,
    'Predicted Revenue for Non-Superhost': np.exp(nonsuperhost_revenue_predict)

})

# Save the DataFrame to a CSV file
results_df.to_csv('predicted_revenue_nonsuperhost.csv', index=False)

import pandas as pd

df = pd.read_csv('predicted_revenue_nonsuperhost.csv')

results_df = pd.DataFrame({
    'Airbnb Host ID': [],
    'Total Actual Revenue per host': [],
    'Total Predicted Revenue per host': [],
    'Predicted - Actual': []
})

total_true = 0
total_predict = 0
Cur_host = 0

for index, row in df.iterrows():
    if Cur_host == 0:
        Cur_host = row['Airbnb Host ID']
        total_true += row['Actual Revenue']
        total_predict += row['Predicted Revenue for Non-Superhost']
    elif row['Airbnb Host ID'] == Cur_host:
        total_true += row['Actual Revenue']
        total_predict += row['Predicted Revenue for Non-Superhost']
    elif row['Airbnb Host ID'] != Cur_host:
        results_df = results_df.append({
            'Airbnb Host ID': Cur_host,
            'Total Actual Revenue per host': total_true,
            'Total Predicted Revenue per host': total_predict,
            'Predicted - Actual': total_predict - total_true
        }, ignore_index=True)
        Cur_host = row['Airbnb Host ID']
        total_true = row['Actual Revenue']
        total_predict = row['Predicted Revenue for Non-Superhost']

# Append the last host data
results_df = results_df.append({
    'Airbnb Host ID': Cur_host,
    'Total Actual Revenue per host': total_true,
    'Total Predicted Revenue per host': total_predict,
    'Predicted - Actual': total_predict - total_true
}, ignore_index=True)




# Save the DataFrame to a CSV file
results_df.to_csv('predicted_revenue_nonsuperhost(per_host).csv', index=False)
