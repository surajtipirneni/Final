
# coding: utf-8

# In[7]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import (
    linear_model, metrics, neural_network, pipeline, model_selection
)

colors = ['#165aa7', '#cb495c', '#fec630', '#bb60d5', '#f47915', '#06ab54', '#002070', '#b27d12', '#007030']

url = "https://datascience.quantecon.org/assets/data/kc_house_data.csv"
df = pd.read_csv(url)
df.info()


# In[6]:

#Excercise 1

#Model
# import

from sklearn import linear_model

# construct the model instance
sqft_lr_model = linear_model.LinearRegression()

# fit the model
sqft_lr_model.fit(X[["sqft_living"]], y)

# print the coefficients
beta_0 = sqft_lr_model.intercept_
beta_1 = sqft_lr_model.coef_[0]

print(f"Fit model: log(price) = {beta_0:.4f} + {beta_1:.4f} sqft_living")


# In[7]:

# Generate predictions for all data points in the sample
y_pred = sqft_lr_model.predict(X[["sqft_living"]])

# Print the predicted values
print(y_pred)


# In[ ]:

# Create scatter plot with actual data
plt.scatter(X["sqft_living"], y, color=colors[0], alpha=0.5, label="Actual")

# Add scatter plot with model predictions
plt.scatter(X["sqft_living"], y_pred, color="red", alpha=0.25, label="Predicted")

# Add labels and legend
plt.xlabel("Square footage of living area")
plt.ylabel("Log of house price")
plt.legend(loc="upper left")

# Show the plot
plt.show()


# In[ ]:

#Excercise 2


# In[ ]:

from sklearn.metrics import mean_squared_error

# Use the fitted linear regression model to make predictions on the test data
y_pred = sqft_lr_model.predict(X_test[["sqft_living"]])

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")


# In[ ]:

# Excercise 3


# In[ ]:

lr_model = linear_model.LinearRegression()
lr_model.fit(X, y)


# In[ ]:

# Mean squared error

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the lr_model on the training data
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train the sqft_lr_model on the training data
sqft_lr_model = LinearRegression()
sqft_lr_model.fit(X_train[["sqft_living"]], y_train)

# Use the lr_model to make predictions on the test data
lr_pred = lr_model.predict(X_test)

# Use the sqft_lr_model to make predictions on the test data
sqft_lr_pred = sqft_lr_model.predict(X_test[["sqft_living"]])

# Calculate the mean squared error for the lr_model and sqft_lr_model
lr_mse = mean_squared_error(y_test, lr_pred)
sqft_lr_mse = mean_squared_error(y_test, sqft_lr_pred)

print(f"Mean Squared Error (lr_model): {lr_mse:.2f}")
print(f"Mean Squared Error (sqft_lr_model): {sqft_lr_mse:.2f}")


# In[ ]:

'''The model with a lower mean squared error is considered to have a better fit, as it indicates that the model 
is better able to predict the target variable on unseen data. 
Therefore, the model with a lower mean squared error between lr_model and sqft_lr_model has a better fit.'''


# In[ ]:

#Excercise 4

'''To improve the fit of the full model by adding additional features created from the existing ones, 
    we can use feature engineering techniques'''

'''In this case, I am using Polynomial Features'''


# In[ ]:

from sklearn.preprocessing import PolynomialFeatures

# Load the data
url = "https://datascience.quantecon.org/assets/data/kc_house_data.csv"
data = pd.read_csv(url)

# Create interaction features
data['sqft_living_lot'] = data['sqft_living'] * data['sqft_lot']

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(data[['sqft_living', 'sqft_lot']])

# Add polynomial features to data
data = pd.concat([data, pd.DataFrame(X_poly, columns=['sqft_living_poly', 'sqft_lot_poly', 'sqft_living_lot_poly'])], axis=1)

# Split the data into training and testing sets
X = data.drop(['price', 'date'], axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model on the updated data
model = LinearRegression()
model.fit(X_train, y_train)

# Use the model to make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (with interaction and polynomial features): {mse:.2f}")


# In[ ]:

#Excercise 5

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on the training set
lr_model = linear_model.LinearRegression()
lr_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = lr_model.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse:.4f}")


# In[ ]:

#Excercise 6

from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Load the Boston housing dataset
boston = load_boston()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)

# Create the decision tree regressor with varying regularization parameters
for ccp_alpha in [0.0, 0.001, 0.01, 0.1]:
    # Create the decision tree regressor with the specified regularization parameter
    dtr = DecisionTreeRegressor(ccp_alpha=ccp_alpha, random_state=42)
    dtr.fit(X_train, y_train)

    # Plot the tree
    plt.figure()
    plot_tree(dtr, filled=True)
    plt.title("Decision Tree Regressor with ccp_alpha", {ccp_alpha:.4f})
    plt.show()

    # Evaluate the model on the test set
    score = dtr.score(X_test, y_test)
    print(f"ccp_alpha={ccp_alpha:.4f}: Test set R^2 score={score:.4f}")


# In[ ]:

#Excercise 7

from sklearn.tree import DecisionTreeRegressor, export_graphviz
import graphviz

# Load the Boston housing dataset
boston = load_boston()

# Create the decision tree regressor
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(boston.data, boston.target)

# Visualize the decision graph using Graphviz
dot_data = export_graphviz(dtr, out_file=None, feature_names=boston.feature_names,
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('boston_housing_tree', format='png')
graph


# In[ ]:

# Excercise 8

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

# Load the Boston housing dataset
boston = load_boston()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)

# Fit a random forest regressor to the training data
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Use the trained random forest model to predict on the test data and calculate MSE
rf_pred = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
print("Random Forest MSE:", rf_mse)

# Fit a Lasso model to the training data
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train, y_train)

# Use the trained Lasso model to predict on the test data and calculate MSE
lasso_pred = lasso.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)
print("Lasso MSE:", lasso_mse)

# Plot the feature importances as a bar chart
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, boston.feature_names[indices[f]], importances[indices[f]]))

# Plot the feature importances
plt.figure(figsize=(10,5))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), boston.feature_names[indices])
plt.xlim([-1, X_train.shape[1]])
plt.show()


# #Excercise 9
# 
# ws = [w1, w2, ..., wend]
# bs = [b1, b2, ..., bend]
# 
# def eval_mlp(X, ws, bs, f):
#     
#     N = len(ws) - 1
# 
#     out = X
#     for i in range(N):
#         out = f(np.dot(out, ws[i]) + bs[i])
# 
# # For this step remember python starts counting at 0!
#     return np.dot(out, ws[N]) + bs[N]
# 

# In[ ]:



