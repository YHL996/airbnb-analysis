import pandas as pd
cities = ["Los Angeles", "Chicago", "Dallas", "Houston", "Miami", "Oakland", "Philadelphia", "Washington", "Boston"]
# "Chicago", "Dallas", "Houston", "Oakland", "Washington", "Boston"
# Assuming your dataset is stored in a CSV file named 'your_dataset.csv'
df = pd.read_csv("airbnb_New York.csv")

# Convert the 'Scraped Date' column to datetime format
df['Scraped Date'] = pd.to_datetime(df['Scraped Date'])

# Filter the data for values between May 2020 and June 2020
filtered_df = df[(df['Scraped Date'] >= '2020-05-01') & (df['Scraped Date'] <= '2020-06-30')]

# Assuming your DataFrame is named df
filtered_df = filtered_df.groupby('Airbnb Host ID').agg({
    'Superhost': 'max',
    'revenue': 'mean',
    'booked_days_avePrice': 'mean',
    'occupancy_rate': 'mean',
    'available_days': 'mean',
    'booked_days': 'mean'
}).reset_index()

from scipy.stats import ttest_ind

# Remove rows with missing or zero revenue values
filtered_df = filtered_df[(filtered_df['revenue'].notnull())]
# filtered_df = filtered_df[(filtered_df['revenue'].notnull()) & (filtered_df['revenue'] != 0)]

# Calculate average revenue for Superhosts and non-Superhosts
superhost_avg_revenue = filtered_df[filtered_df['Superhost'] == 1]['revenue'].mean()
non_superhost_avg_revenue = filtered_df[filtered_df['Superhost'] == 0]['revenue'].mean()

print(f'Average Revenue for Superhosts: ${superhost_avg_revenue:.2f}')
print(f'Average Revenue for Non-Superhosts: ${non_superhost_avg_revenue:.2f}')

# Separate revenue for Superhosts and non-Superhosts
superhost_revenue = filtered_df[filtered_df['Superhost'] == 1]['revenue']
non_superhost_revenue = filtered_df[filtered_df['Superhost'] == 0]['revenue']

# Perform t-test
t_statistic, p_value = ttest_ind(superhost_revenue, non_superhost_revenue, equal_var=False, nan_policy='omit')

# Check the p-value
alpha = 0.05
print(f'Test Statistic: {t_statistic:.4f}')
print(f'P-value: {p_value:.4f}')

# Check for significance
if p_value < alpha:
    print("The difference in revenue between Superhosts and non-Superhosts is statistically significant.")
else:
    print("There is no significant difference in revenue between Superhosts and non-Superhosts.")


from scipy.stats import ttest_ind

# Filter the data for values between May 2020 and June 2020
filtered_df = df[(df['Scraped Date'] >= '2020-05-01') & (df['Scraped Date'] <= '2020-06-30')]

filtered_df = filtered_df.groupby('Airbnb Host ID').agg({
    'Superhost': 'max',
    'revenue': 'mean',
    'booked_days_avePrice': 'mean',
    'occupancy_rate': 'mean',
    'available_days': 'mean',
    'booked_days': 'mean'
}).reset_index()

# Remove rows with missing or zero values in the 'booked_days_avePrice' column
filtered_df = filtered_df[(filtered_df['booked_days_avePrice'].notnull()) & (filtered_df['booked_days_avePrice'] != 0)]

# Separate booked days' average price for Superhosts and non-Superhosts
superhost_avg_price = filtered_df[filtered_df['Superhost'] == 1]['booked_days_avePrice']
non_superhost_avg_price = filtered_df[filtered_df['Superhost'] == 0]['booked_days_avePrice']

# Print out the average booked days' average price for both Superhosts and non-Superhosts
print(f'Average Price for Superhosts: ${superhost_avg_price.mean():.2f}')
print(f'Average Price for Non-Superhosts: ${non_superhost_avg_price.mean():.2f}')

# Perform t-test
t_statistic, p_value = ttest_ind(superhost_avg_price, non_superhost_avg_price, equal_var=False, nan_policy='omit')

# Check the p-value
alpha = 0.05
print(f'\nTest Statistic: {t_statistic:.4f}')
print(f'P-value: {p_value:.4f}')

# Check for significance
if p_value < alpha:
    print("The difference in average price between Superhosts and non-Superhosts is statistically significant.")
else:
    print("There is no significant difference in average price between Superhosts and non-Superhosts.")

from scipy.stats import ttest_ind

# Filter the data for values between May 2020 and June 2020
filtered_df = df[(df['Scraped Date'] >= '2020-05-01') & (df['Scraped Date'] <= '2020-06-30')]

filtered_df = filtered_df.groupby('Airbnb Host ID').agg({
    'Superhost': 'max',
    'revenue': 'mean',
    'booked_days_avePrice': 'mean',
    'occupancy_rate': 'mean',
    'available_days': 'mean',
    'booked_days': 'mean'
}).reset_index()

# Remove rows with missing or zero values in the relevant columns
filtered_df = filtered_df[(filtered_df['occupancy_rate'].notnull()) & (filtered_df['occupancy_rate'] != 0)]
filtered_df = filtered_df[(filtered_df['available_days'].notnull()) & (filtered_df['available_days'] != 0)]
filtered_df = filtered_df[(filtered_df['booked_days'].notnull()) & (filtered_df['booked_days'] != 0)]

# Separate data for Superhosts and non-Superhosts
superhost_occupancy_rate = filtered_df[filtered_df['Superhost'] == 1]['occupancy_rate']
non_superhost_occupancy_rate = filtered_df[filtered_df['Superhost'] == 0]['occupancy_rate']

superhost_available_days = filtered_df[filtered_df['Superhost'] == 1]['available_days']
non_superhost_available_days = filtered_df[filtered_df['Superhost'] == 0]['available_days']

superhost_booked_days = filtered_df[filtered_df['Superhost'] == 1]['booked_days']
non_superhost_booked_days = filtered_df[filtered_df['Superhost'] == 0]['booked_days']

# Print out the statistics
print(f'Occupancy Rate for Superhosts: Mean={superhost_occupancy_rate.mean():.4f}, Std={superhost_occupancy_rate.std():.4f}')
print(f'Occupancy Rate for Non-Superhosts: Mean={non_superhost_occupancy_rate.mean():.4f}, Std={non_superhost_occupancy_rate.std():.4f}')

print(f'\nAvailable Days for Superhosts: Mean={superhost_available_days.mean():.4f}, Std={superhost_available_days.std():.4f}')
print(f'Available Days for Non-Superhosts: Mean={non_superhost_available_days.mean():.4f}, Std={non_superhost_available_days.std():.4f}')

print(f'\nBooked Days for Superhosts: Mean={superhost_booked_days.mean():.4f}, Std={superhost_booked_days.std():.4f}')
print(f'Booked Days for Non-Superhosts: Mean={non_superhost_booked_days.mean():.4f}, Std={non_superhost_booked_days.std():.4f}')

# Perform t-tests
t_stat_occ, p_value_occ = ttest_ind(superhost_occupancy_rate, non_superhost_occupancy_rate, equal_var=False, nan_policy='omit')
t_stat_avail, p_value_avail = ttest_ind(superhost_available_days, non_superhost_available_days, equal_var=False, nan_policy='omit')
t_stat_booked, p_value_booked = ttest_ind(superhost_booked_days, non_superhost_booked_days, equal_var=False, nan_policy='omit')

# Check the p-values
alpha = 0.05
print(f'\nOccupancy Rate Hypothesis Test: Test Statistic={t_stat_occ:.4f}, P-value={p_value_occ:.4f}')
print(f'Available Days Hypothesis Test: Test Statistic={t_stat_avail:.4f}, P-value={p_value_avail:.4f}')
print(f'Booked Days Hypothesis Test: Test Statistic={t_stat_booked:.4f}, P-value={p_value_booked:.4f}')

# Check for significance
if p_value_occ < alpha:
    print("The difference in occupancy rate between Superhosts and non-Superhosts is statistically significant.")
else:
    print("There is no significant difference in occupancy rate between Superhosts and non-Superhosts.")

if p_value_avail < alpha:
    print("The difference in available days between Superhosts and non-Superhosts is statistically significant.")
else:
    print("There is no significant difference in available days between Superhosts and non-Superhosts.")

if p_value_booked < alpha:
    print("The difference in booked days between Superhosts and non-Superhosts is statistically significant.")
else:
    print("There is no significant difference in booked days between Superhosts and non-Superhosts.")
