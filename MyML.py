import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import statistics
from scipy.stats import shapiro
import seaborn as sns

##                                                             Data read and preprocessing

data = pd.read_csv(r'C:\Users\milosz.groch\Desktop\Kaggle\Simple ML\housing.csv')
data.info()

# We can see that one column is an object so we want to drop it from our data
data = data.drop('ocean_proximity', axis=1)

# Let's check the NA's
missings = data.isna().sum()
print(missings)

# We have 207 NA's in "total bedrooms" column. We could drop the data that contains NA's but better approach is to impute the missing values. We will use the median for total_bedrooms
median = statistics.median(data['total_bedrooms'])
data['total_bedrooms'].fillna(median, inplace=True)


##                                                              EXPLORATORY DATA ANALYSIS

# Now we wants to check:
# A) Data distribution (For example lineal regression needs normal distribution)
# B) Outliers
# C) Data abnormalities


# To check that let's do:

# 1) Histogram
data.hist(figsize=(20,10), bins=40)
plt.show()


# 2) Shapiro test if all data is normally distributed
norm = data.apply(lambda x: shapiro(x))
print(norm)
# No values under p=0.05 so every column has normal distribution

# 3) Correlation matrix
correlation = data.corr()
print(correlation)
sns.heatmap(correlation, annot=True)
plt.show()
# The only significantly correlated variable with median_house_value is median_income. We have some low correlation variables but we'll leave them for now. We can experiment with them later

# We noticed in the previous histogram that values in the housing_median_age and median_house_value columns look like outliers so we will drop them
max_hous = data['housing_median_age'].max()
max_med = data['median_house_value'].max()

data = data[data['housing_median_age'] != max_hous]
data = data[data['median_house_value'] != max_med]


##                                                                 Data Splitting

from sklearn.model_selection import train_test_split

y = data['median_house_value']
x = data.drop('median_house_value', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1000)


##                                                                 Model Building and Performance

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge

from sklearn.metrics import mean_squared_error,r2_score

models = {'rf': RandomForestRegressor(), 'lr': LinearRegression(), 'svr': SVR(), 'XGB': XGBRegressor(), 'Ridge': Ridge(), 'en': ElasticNet(), 'SGD': SGDRegressor(), 'Bayesian': BayesianRidge()}
results = []

for name, model in models.items():
    model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    MSE_train = mean_squared_error(y_train, y_train_pred)
    R2_train = r2_score(y_train, y_train_pred)
    MSE_test = mean_squared_error(y_test, y_test_pred)
    R2_test = r2_score(y_test, y_test_pred)
    results.append([name, MSE_train, R2_train, MSE_test, R2_test])

results_df = pd.DataFrame(results, columns=['Model','MSE_train','R2_train','MSE_test','R2_test'])

results_df = results_df.round(2)
results_df = results_df.applymap(lambda x: '{:,.2f}'.format(x) if isinstance(x, (float, float)) else x)
print(results_df)

#                      RF


RF = RandomForestRegressor(max_depth=10, random_state=100)
RF.fit(x_train, y_train)

y_train_pred_rf = RF.predict(x_train)
y_test_pred_rf = RF.predict(x_test)


#                       LR


LR = LinearRegression()
LR.fit(x_train, y_train)

y_train_pred_lr = LR.predict(x_train)
y_test_pred_lr = LR.predict(x_test)


#                      SVR


SVR = SVR()
SVR.fit(x_train, y_train)

y_train_pred_svr = SVR.predict(x_train)
y_test_pred_svr = SVR.predict(x_test)


#                       XGB


XGB = XGBRegressor()
XGB.fit(x_train, y_train)

y_train_pred_xgb = XGB.predict(x_train)
y_test_pred_xgb = XGB.predict(x_test)


#                      Ridge


RIDGE = Ridge()
RIDGE.fit(x_train, y_train)

y_train_pred_ridge = RIDGE.predict(x_train)
y_test_pred_ridge = RIDGE.predict(x_test)


#                       ElasticNet


EN = ElasticNet()
EN.fit(x_train, y_train)

y_train_pred_en = EN.predict(x_train)
y_test_pred_en = EN.predict(x_test)


#                      SGD


SGD = SGDRegressor()
SGD.fit(x_train, y_train)

y_train_pred_sgd = SGD.predict(x_train)
y_test_pred_sgd = SGD.predict(x_test)


#                       Bayesian


BAY = BayesianRidge()
BAY.fit(x_train, y_train)

y_train_pred_bay = BAY.predict(x_train)
y_test_pred_bay = BAY.predict(x_test)




##                                                                 Model Performance

from sklearn.metrics import mean_squared_error,r2_score

MSE_y_train_rf = mean_squared_error(y_train, y_train_pred_rf)
R2_y_train_rf = r2_score(y_train, y_train_pred_rf)

MSE_y_test_rf = mean_squared_error(y_test, y_test_pred_rf)
R2_y_test_rf = r2_score(y_test, y_test_pred_rf)


results_rf = pd.DataFrame(['rf', MSE_y_train_rf, R2_y_train_rf, MSE_y_test_rf, R2_y_test_rf]).transpose()
results_rf.columns = ['Method',' MSE_y_train_rf', 'R2_y_train_rf', 'MSE_y_test_rf', 'R2_y_test_rf']

results_rf