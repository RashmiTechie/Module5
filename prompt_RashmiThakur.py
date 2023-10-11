#!/usr/bin/env python
# coding: utf-8

# ### Will a Customer Accept the Coupon?
# 
# **Context**
# 
# Imagine driving through town and a coupon is delivered to your cell phone for a restaraunt near where you are driving. Would you accept that coupon and take a short detour to the restaraunt? Would you accept the coupon but use it on a sunbsequent trip? Would you ignore the coupon entirely? What if the coupon was for a bar instead of a restaraunt? What about a coffee house? Would you accept a bar coupon with a minor passenger in the car? What about if it was just you and your partner in the car? Would weather impact the rate of acceptance? What about the time of day?
# 
# Obviously, proximity to the business is a factor on whether the coupon is delivered to the driver or not, but what are the factors that determine whether a driver accepts the coupon once it is delivered to them? How would you determine whether a driver is likely to accept a coupon?
# 
# **Overview**
# 
# The goal of this project is to use what you know about visualizations and probability distributions to distinguish between customers who accepted a driving coupon versus those that did not.
# 
# **Data**
# 
# This data comes to us from the UCI Machine Learning repository and was collected via a survey on Amazon Mechanical Turk. The survey describes different driving scenarios including the destination, current time, weather, passenger, etc., and then ask the person whether he will accept the coupon if he is the driver. Answers that the user will drive there ‘right away’ or ‘later before the coupon expires’ are labeled as ‘Y = 1’ and answers ‘no, I do not want the coupon’ are labeled as ‘Y = 0’.  There are five different types of coupons -- less expensive restaurants (under \\$20), coffee houses, carry out & take away, bar, and more expensive restaurants (\\$20 - \\$50). 

# **Deliverables**
# 
# Your final product should be a brief report that highlights the differences between customers who did and did not accept the coupons.  To explore the data you will utilize your knowledge of plotting, statistical summaries, and visualization using Python. You will publish your findings in a public facing github repository as your first portfolio piece. 
# 
# 
# 
# 

# ### Data Description
# Keep in mind that these values mentioned below are average values.
# 
# The attributes of this data set include:
# 1. User attributes
#     -  Gender: male, female
#     -  Age: below 21, 21 to 25, 26 to 30, etc.
#     -  Marital Status: single, married partner, unmarried partner, or widowed
#     -  Number of children: 0, 1, or more than 1
#     -  Education: high school, bachelors degree, associates degree, or graduate degree
#     -  Occupation: architecture & engineering, business & financial, etc.
#     -  Annual income: less than \\$12500, \\$12500 - \\$24999, \\$25000 - \\$37499, etc.
#     -  Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
#     -  Number of times that he/she buys takeaway food: 0, less than 1, 1 to 3, 4 to 8 or greater
#     than 8
#     -  Number of times that he/she goes to a coffee house: 0, less than 1, 1 to 3, 4 to 8 or
#     greater than 8
#     -  Number of times that he/she eats at a restaurant with average expense less than \\$20 per
#     person: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
#     -  Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
#     
# 
# 2. Contextual attributes
#     - Driving destination: home, work, or no urgent destination
#     - Location of user, coupon and destination: we provide a map to show the geographical
#     location of the user, destination, and the venue, and we mark the distance between each
#     two places with time of driving. The user can see whether the venue is in the same
#     direction as the destination.
#     - Weather: sunny, rainy, or snowy
#     - Temperature: 30F, 55F, or 80F
#     - Time: 10AM, 2PM, or 6PM
#     - Passenger: alone, partner, kid(s), or friend(s)
# 
# 
# 3. Coupon attributes
#     - time before it expires: 2 hours or one day

# In[255]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# ### Problems
# 
# Use the prompts below to get started with your data analysis.  
# 
# 1. Read in the `coupons.csv` file.
# 
# 
# 

# In[256]:


data = pd.read_csv('data/coupons.csv')


# In[257]:


data.head()


# 2. Investigate the dataset for missing or problematic data.

# In[258]:


print(data.isnull().sum())
print("Number of duplicate rows: ", data.duplicated().sum())


# 3. Decide what to do about your missing data -- drop, replace, other...

# In[259]:


data.fillna(data.mean(), inplace=True)
data = data.dropna()
data = data.drop_duplicates()


# 4. What proportion of the total observations chose to accept the coupon? 
# 
# 

# In[260]:


total_observations = data.shape[0]
accepted_observations = data[data['Y'] == 1].shape[0]
proportion_accepted = accepted_observations / total_observations

print(f"Proportion of total observations that accepted the coupon: {proportion_accepted:.2f}")


# 5. Use a bar plot to visualize the `coupon` column.

# In[261]:


coupon_counts = data['coupon'].value_counts()

# Create a bar plot
plt.figure(figsize=(10, 7))
sns.barplot(x=coupon_counts.index, y=coupon_counts.values)

# Add labels and title
plt.xlabel('Type of Coupon')
plt.ylabel('Count')
plt.title('Count of Each Type of Coupon')

# Display the plot
plt.xticks(rotation=45) 
plt.show()







# 6. Use a histogram to visualize the temperature column.

# In[262]:


plt.figure(figsize=(10, 7))
sns.histplot(data['temperature'], bins=30, kde=True)  # bins determines the number of bins, and kde=True adds a Kernel Density Estimate plot

# Add labels and title
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.title('Distribution of Temperatures')

# Display the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# **Investigating the Bar Coupons**
# 
# Now, we will lead you through an exploration of just the bar related coupons.  
# 
# 1. Create a new `DataFrame` that contains just the bar coupons.
# 

# In[263]:


bar_coupons_df = data.loc[data['coupon'] == 'Bar']
print(bar_coupons_df.head())
                          


# 2. What proportion of bar coupons were accepted?
# 

# In[264]:


total_observations = bar_coupons_df.shape[0]
accepted_observations = bar_coupons_df[bar_coupons_df['Y'] == 1].shape[0]
proportion_accepted = accepted_observations / total_observations

print(f"Proportion of bar coupons that accepted the coupon: {proportion_accepted:.2f}")


# 3. Compare the acceptance rate between those who went to a bar 3 or fewer times a month to those who went more.
# 

# In[265]:


#less_frequent = bar_coupons_df[bar_coupons_df['Bar'] == '1~3']

less_frequent = bar_coupons_df[bar_coupons_df['Bar'] == '1~3']

more_frequent = bar_coupons_df[~(bar_coupons_df['Bar'] == '1~3')]

acceptance_rate_less_frequent = less_frequent['Y'].mean()

# Calculate acceptance rate for those who go to a bar more than 3 times a month
acceptance_rate_more_frequent = more_frequent['Y'].mean()

# Print out the results
print(f"Acceptance rate for those who go to a bar 3 or fewer times a month: {acceptance_rate_less_frequent:.2f}")
print(f"Acceptance rate for those who go to a bar more than 3 times a month: {acceptance_rate_more_frequent:.2f}")




# 4. Compare the acceptance rate between drivers who go to a bar more than once a month and are over the age of 25 to the all others.  Is there a difference?
# 

# In[266]:


# Convert 'age' column to integers
bar_coupons_df['age'] = pd.to_numeric(bar_coupons_df['age'], errors='coerce')  

# Drivers over 25 who go to a bar more than once a month
group1 = bar_coupons_df[(bar_coupons_df['age'] > 25) & (~(bar_coupons_df['Bar'] == 'less1') | (bar_coupons_df['Bar'] == 'never'))]

# All other drivers

group2 = bar_coupons_df[~((bar_coupons_df['age'] > 25) & (~(bar_coupons_df['Bar'] == 'less1') | (bar_coupons_df['Bar'] == 'never')))]


# Acceptance rate for all other drivers
acceptance_rate_group1 = group1['Y'].mean()


# Acceptance rate for drivers over 25 who go to a bar more than once a month

acceptance_rate_group2 = group2['Y'].mean()

# Print the acceptance rates
print(f"Acceptance rate for drivers over 25 who go to a bar more than once a month: {acceptance_rate_group1:.2f}")
print(f"Acceptance rate for all other drivers: {acceptance_rate_group2:.2f}")


# 5. Use the same process to compare the acceptance rate between drivers who go to bars more than once a month and had passengers that were not a kid and had occupations other than farming, fishing, or forestry. 
# 

# In[267]:


# Drivers who go to bars more than once a month, had passengers that were not a kid, and had occupations other than farming, fishing, or forestry

group1 = bar_coupons_df[(bar_coupons_df['passanger'] !='kid') & 
                        (~(bar_coupons_df['Bar'] == 'less1') | (bar_coupons_df['Bar'] == 'never')) &
                       (~bar_coupons_df['occupation'].isin(['farming', 'fishing', 'forestry']))]

# All other drivers
group2 =  bar_coupons_df[~((bar_coupons_df['passanger'] !='kid') & 
                        (~(bar_coupons_df['Bar'] == 'less1') | (bar_coupons_df['Bar'] == 'never')) &
                        (~bar_coupons_df['occupation'].isin(['farming', 'fishing', 'forestry'])))]

# Calculate acceptance rates
acceptance_rate_group1 = group1['Y'].mean()
acceptance_rate_group2 = group2['Y'].mean()

# Print the acceptance rates
print(f"Acceptance rate for drivers who go to bars more than once a month, had passengers that were not a kid, and had occupations other than farming, fishing, or forestry: {acceptance_rate_group1:.2f}")
print(f"Acceptance rate for all other drivers: {acceptance_rate_group2:.2f}")

# Compare the acceptance rates
difference = acceptance_rate_group1 - acceptance_rate_group2
print(f"Difference in acceptance rates: {difference:.2f}")

if difference > 0:
    print("The specified group has a higher acceptance rate.")
elif difference < 0:
    print("All other drivers have a higher acceptance rate.")
else:
    print("Both groups have the same acceptance rate.")



# 6. Compare the acceptance rates between those drivers who:
# 
# - go to bars more than once a month, had passengers that were not a kid, and were not widowed *OR*
# - go to bars more than once a month and are under the age of 30 *OR*
# - go to cheap restaurants more than 4 times a month and income is less than 50K. 
# 
# 

# In[268]:


# Group 1: Go to bars more than once a month, had passengers that were not a kid, and were not widowed

group1 = bar_coupons_df[(bar_coupons_df['passanger'] !='kid') & 
                        (~(bar_coupons_df['Bar'] == 'less1') | (bar_coupons_df['Bar'] == 'never')) &
                        (bar_coupons_df['maritalStatus']!='widowed')]

# Group 2: Go to bars more than once a month and are under the age of 30
group2 = bar_coupons_df[(bar_coupons_df['age'] < 30) & 
                        (~(bar_coupons_df['Bar'] == 'less1') | (bar_coupons_df['Bar'] == 'never'))]

# Group 3: Go to cheap restaurants more than 4 times a month and income is less than 50K

#Restaurant coupon dataset 
rest_coupons_df = data.loc[data['coupon'].isin(['Restaurant(20-50)','Restaurant(<20)'])]

# Using the lower bound of the income range
rest_coupons_df['income'] = rest_coupons_df['income'].str.extract('(\d+)').astype(int)

group3 = rest_coupons_df[rest_coupons_df['income'] < 50000]


# Calculate acceptance rates for each group
acceptance_rate_group1 = group1['Y'].mean()
acceptance_rate_group2 = group2['Y'].mean()
acceptance_rate_group3 = group3['Y'].mean()

# Print the acceptance rates
print(f"Acceptance rate for Group 1: {acceptance_rate_group1:.2f}")
print(f"Acceptance rate for Group 2: {acceptance_rate_group2:.2f}")
print(f"Acceptance rate for Group 3: {acceptance_rate_group3:.2f}")

# Compare the acceptance rates
# Note: For a more detailed comparison, you can perform statistical tests to evaluate if the differences in acceptance rates are statistically significant.
# Compare the acceptance rates
difference = acceptance_rate_group1 - acceptance_rate_group2
print(f"Difference in acceptance rates: {difference:.2f}")

if difference > 0:
    print("The specified group has a higher acceptance rate.")
elif difference < 0:
    print("Drivers who goes to bars more than once a month and are under the age of 30 have a higher acceptance rate.")
else:
    print("Both groups have the same acceptance rate.")

# Compare the acceptance rates
difference = acceptance_rate_group2 - acceptance_rate_group3
print(f"Difference in acceptance rates: {difference:.2f}")

if difference > 0:
    print("Drivers who goes to bars more than once a month and are under the age of 30 have a higher acceptance rate.")
elif difference < 0:
    print("Drivers goes to restaurant have a higher acceptance rate.")
else:
    print("Both groups have the same acceptance rate.")





# 7.  Based on these observations, what do you hypothesize about drivers who accepted the bar coupons?

# In[269]:


#Plot the Frequency of Going to Bars
plt.figure(figsize=(10, 7))
sns.countplot(data=bar_coupons_df, x='Bar', hue='Y', palette='viridis')
plt.title('Frequency of Going to Bars vs Coupon Acceptance')
plt.xlabel('Number of Times Going to Bars per Month')
plt.ylabel('Count')
plt.legend(title='Coupon Accepted', loc='upper right')
plt.show()

print("Thsi plot shows that freuency of number of times going to the bar after accepting coupon is greater than 8")

# Filter data for those who go to bars more than once a month
bar_goers = bar_coupons_df[bar_coupons_df['Bar'].isin(['1~3','4-8','gt8'])]

plt.figure(figsize=(10, 4))
sns.barplot(data=bar_goers, x='age', y='Y', ci=None, palette='mako')
plt.title('Acceptance Rate by Age for Drivers Who Go to Bars More Than Once a Month')
plt.xlabel('Age')
plt.ylabel('Acceptance Rate')
plt.show()


print("Thsi plot shows that younger drivers of age 26 are more tend to go to the bar")




# ### Independent Investigation
# 
# Using the bar coupon example as motivation, you are to explore one of the other coupon groups and try to determine the characteristics of passengers who accept the coupons.  

# In[270]:


# Coffee coupon dataset Filter Data for Coffee House Coupons
coffee_coupons_df = data.loc[data['coupon']=='Coffee House']


# In[271]:


# Visualize Coupon Acceptance Rate by Gender
plt.figure(figsize=(8, 5))
sns.countplot(data=coffee_coupons_df, x='gender', hue='Y', palette='viridis')
plt.title('Coffee House Coupon Acceptance by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Coupon Accepted', loc='upper right')
plt.show()


# In[272]:


# Visualize Coupon Acceptance Rate by Income
# Convert income to numeric using lower bound
coffee_coupons_df = data.loc[data['coupon']=='Coffee House']


coffee_coupons_df['income'] = coffee_coupons_df['income'].str.extract('(\d+)').astype(float)


plt.figure(figsize=(10, 7))
sns.barplot(data=coffee_coupons_df, x='income', y='Y', ci=None, palette='coolwarm')
plt.title('Coffee House Coupon Acceptance by Income')
plt.xlabel('Income Lower Bound')
plt.ylabel('Acceptance Rate')
plt.show()





# In[273]:


# Visualize Coupon Acceptance Rate by Age
plt.figure(figsize=(8, 5))
sns.countplot(data=coffee_coupons_df, x='age', hue='Y', palette='viridis')
plt.title('Coffee House Coupon Acceptance by Age')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Coupon Accepted', loc='upper right')
plt.show()

print("This plot clearly depicts that coupon acceptance is higher for younger age ")


# In[274]:


#Proportion of bar coupons that accepted 
total_observations = coffee_coupons_df.shape[0]
accepted_observations = coffee_coupons_df[coffee_coupons_df['Y'] == 1].shape[0]
proportion_accepted = accepted_observations / total_observations

print(f"Proportion of Coffee coupons that accepted the coupon: {proportion_accepted:.2f}")


# In[275]:


##Compare the acceptance rate between drivers who go to a Coffee house  more than once a month and are over the age of 25 to the all others.  


# In[276]:


# Convert 'age' column to integers
coffee_coupons_df['age'] = pd.to_numeric(coffee_coupons_df['age'], errors='coerce')  

# Drivers over 25 who go to a bar more than once a month
group1 = coffee_coupons_df[(coffee_coupons_df['age'] > 25) & (~(coffee_coupons_df['CoffeeHouse'] == 'less1') | (coffee_coupons_df['CoffeeHouse'] == 'never'))]

# All other drivers

group2 = coffee_coupons_df[~((coffee_coupons_df['age'] > 25) & (~(coffee_coupons_df['CoffeeHouse'] == 'less1') | (coffee_coupons_df['CoffeeHouse'] == 'never')))]


# Acceptance rate for all other drivers
acceptance_rate_group1 = group1['Y'].mean()


# Acceptance rate for drivers over 25 who go to a bar more than once a month

acceptance_rate_group2 = group2['Y'].mean()

# Print the acceptance rates
print(f"Acceptance rate for drivers over 25 who go to a CoffeeHouse more than once a month: {acceptance_rate_group1:.2f}")
print(f"Acceptance rate for all other drivers: {acceptance_rate_group2:.2f}")

# Compare the acceptance rates
difference = acceptance_rate_group1 - acceptance_rate_group2
print(f"Difference in acceptance rates: {difference:.2f}")

if difference > 0:
    print("Acceptance rate for drivers over 25 who go to a CoffeeHouse more than once a month has a higher acceptance rate.")
elif difference < 0:
    print("All other drivers have a higher acceptance rate.")
else:
    print("Both groups have the same acceptance rate.")


# In[277]:


# Visualize Coupon Acceptance Rate by temperature
plt.figure(figsize=(8, 5))
sns.countplot(data=coffee_coupons_df, x='temperature', hue='Y', palette='viridis')
plt.title('Coffee House Coupon Acceptance by Temperature')
plt.xlabel('Temp')
plt.ylabel('Count')
plt.legend(title='Coupon Accepted', loc='upper right')
plt.show()

print("This plot clearly shows that coupon acceptance is higher for Temperature more than 55 ")


# In[ ]:




