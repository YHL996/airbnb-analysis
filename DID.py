#DID(revenue-treatemnt-gain_groupby)
import pandas as pd
import statsmodels.api as sm
import numpy as np

cities = ["Los Angeles", "Chicago", "Dallas", "Houston", "Miami", "Oakland", "Philadelphia", "Washington", "Boston"]
# "Chicago", "Los Angeles", "Chicago", "Dallas", "Houston", "Miami", "Oakland", "Philadelphia", "Washington", "Boston"
# Assuming your dataset is stored in a CSV file named 'your_dataset.csv'
df = pd.read_csv("airbnb_New York.csv")

for city in cities:
    df1 = pd.read_csv(f"airbnb_{city}.csv")
    df = pd.concat([df, df1], ignore_index=True)

# Drop 'Integrated Property Manager' from the list of columns
df = df.drop(['Integrated Property Manager', 'City_y', 'Airbnb Property ID', 'scrapes_in_period', 'Zipcode', 'Latitude', 'Longitude', 'Neighborhood', 'time_to_date_mean', 'prev_time_to_date_mean'], axis=1)
# 'revenue', 'occupancy_rate', 'booked_days_avePrice'
df = df[['Scraped Date','Airbnb Host ID', 'superhost_change_gain_superhost', 'revenue']]

# Filter the data for values between May 2020 and June 2020
df = df[(df['Scraped Date'] >= '2020-02-01') & (df['Scraped Date'] <= '2020-06-30')]

df = df.dropna()

# Create a new variable 'time'
df['After'] = 0  # Initialize with 0
df.loc[(df['Scraped Date'] >= '2020-05-01') & (df['Scraped Date'] <= '2020-06-30'), 'After'] = 1

# Create binary variables for 'Superhost' and interaction term
df['superhost_change_gain_superhost'] = df['superhost_change_gain_superhost'].astype(int)  # Ensure 'Superhost' is numeric

# Assuming df is your DataFrame
df['Treatment(Gain Host)'] = df.apply(lambda row: 1 if row['After'] == 1 and row['superhost_change_gain_superhost'] == 1 else 0, axis=1)

df.to_csv("1234.csv")
df['Airbnb Host ID'] = df['Airbnb Host ID'].astype(int)
df['Treatment(Gain Host)'] = ((df.groupby('Airbnb Host ID')['Treatment(Gain Host)'].transform('max') == 1)).astype(int)


# Convert the Treatment column to 0 or 1
df['Treatment(Gain Host)'] = (df['Treatment(Gain Host)'] == 1).astype(int)

# Find Airbnb Host IDs that appear only once
morethantwo_occurrence_ids = df['Airbnb Host ID'][df['Airbnb Host ID'].duplicated(keep=False) == True]

# Filter the DataFrame to keep only rows with Airbnb Host IDs that have more than one occurrence
df = df[df['Airbnb Host ID'].isin(morethantwo_occurrence_ids)]




df = df.groupby(['Airbnb Host ID', 'After']).agg({
    'revenue': 'sum',
    'superhost_change_gain_superhost': 'max',
    'Treatment(Gain Host)': 'max'
}).reset_index()

df['Treatment(Gain Host)_After'] = df['Treatment(Gain Host)'] * df['After']

# Find Airbnb Host IDs that appear only once
morethantwo_occurrence_ids = df['Airbnb Host ID'][df['Airbnb Host ID'].duplicated(keep=False) == True]

# Filter the DataFrame to keep only rows with Airbnb Host IDs that have more than one occurrence
df = df[df['Airbnb Host ID'].isin(morethantwo_occurrence_ids)]

df.to_csv("1234_2.csv")

# Define independent variables (X) and the target variable (y)
X_did = df[['After', 'Treatment(Gain Host)', 'Treatment(Gain Host)_After']]
X_did = sm.add_constant(X_did)  # Add a constant term
y_did = np.log(df['revenue'])

# Fit the DiD regression model
model_did = sm.OLS(y_did, X_did).fit()

# Display DiD model summary
print(model_did.summary())