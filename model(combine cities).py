import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import statsmodels.api as sm
import matplotlib.pyplot as plt

# cities = ["Chicago", "Dallas", "Houston", "Los Angeles", "Miami", "New York", "Oakland", "Philadelphia", "Washington"]
cities = ["Chicago", "Dallas", "Houston"]
# Assuming your dataset is stored in a CSV file named 'your_dataset.csv'
df = pd.read_csv("airbnb_Boston.csv")

for city in cities:
    df1 = pd.read_csv(f"airbnb_{city}.csv")
    df = pd.concat([df, df1], ignore_index=True)

df.to_csv("combined_data.csv")

# Convert the 'Scraped Date' column to datetime format
df['Scraped Date'] = pd.to_datetime(df['Scraped Date'])

# Filter the data for values between May 2020 and June 2020
filtered_df = df[(df['Scraped Date'] >= '2020-05-01') & (df['Scraped Date'] <= '2020-06-30')]

# Keep only unique samples based on 'Airbnb Host ID'
# filtered_df = filtered_df.drop_duplicates(subset='Airbnb Host ID', keep='first')

# Selecting relevant features for logistic regression
numerical_features = ['rating_ave_pastYear', 'numReviews_pastYear', 'numCancel_pastYear', 'num_5_star_Rev_pastYear', 'prop_5_StarReviews_pastYear', 'hostResponseNumber_pastYear', 'hostResponseAverage_pastYear', 'Bedrooms', 'Bathrooms']
#'tract_total_pop', 'tract_white_perc', 'tract_black_perc', 'tract_asian_perc', 'tract_housing_units' ,'revenue', 'occupancy_rate'
categorical_feature = ['City_x']
#'City_y'

# One-hot encode the categorical feature
categorical_encoded = pd.get_dummies(filtered_df[categorical_feature], prefix=categorical_feature)

# Combine numerical and one-hot encoded categorical features
X = pd.concat([filtered_df[numerical_features], categorical_encoded], axis=1)
y = filtered_df['Superhost']

# Drop rows with missing values
X = X.dropna()
y = y[X.index]  # Align y with the remaining rows in X

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing the logistic regression model
model = LogisticRegression(max_iter=1000)

# Fitting the model on the training data
model.fit(X_train, y_train)

# Using statsmodels to get summary
X_train_with_intercept = sm.add_constant(X_train)  # Add intercept term
log_reg_sm = sm.Logit(y_train, X_train_with_intercept)
result = log_reg_sm.fit()

# Display summary
print(result.summary())


# Making predictions on the testing data
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print(f'Confusion Matrix:\n{conf_matrix}')

# Classification report
classification_rep = classification_report(y_test, predictions)
print(f'Classification Report:\n{classification_rep}')

# Calculate AUC
y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
auc = roc_auc_score(y_test, y_proba)
print(f'AUC: {auc}')

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
