# Project
Comparing Linear Regression, Random Forest and Tensorflow with Keras on Rental Price Prediction
#!/usr/bin/env python
# coding: utf-8

# # Importing Necessary Libraries
# 

# In[1]:


# import libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# # Load Data
# 
# The excel data is loaded into a dataframe and the first 5 rows are displayed

# In[2]:


# load data into dataframe 
rental = pd.read_csv("C://Dataset_House_Rent.csv", index_col=0, low_memory=False)
rental.head()


# In[3]:


# check the summary info of the dataframe
rental.info()


# ------------------------------------------------------------------------------------------------------
# # Preprocessing the data
# 
# <u>Preprocessing</u> the dataset is essential for machine learning algorithms because it helps improve the quality and suitability of the data for modeling. 

# ---------------------------------------------------------------
# ## Drop Missing Values

# In[4]:


# check for missing values
rental.isnull().sum()


# In[5]:


# drop missing values
rental = rental.dropna()
rental = rental.reset_index(drop=True)


# In[6]:


# check for missing values again
rental.isnull().sum()


# -----------------------
# 
# ## Drop Duplicate Values

# In[7]:


print("There are {} duplicate values.".format(rental.duplicated().sum()))
rental[rental.duplicated(keep=False)].head(10)


# In[8]:


# remove duplicate values
rental = rental.drop_duplicates()
rental = rental.reset_index(drop=True)


# In[9]:


# check for duplicate values after removing duplicates
print("There are {} duplicate values.".format(rental.duplicated().sum()))
rental.head()


# ---------------------------------
# 
# ## Clean Data

# In[10]:


# select columns that we want to work with
rental = rental[["BHK", "Rent", "Size", "Floor", "Area Type", "City", "Furnishing Status", "Bathroom"]]
rental.head(10)


# In[11]:


# check for typo or wrong spelling for City column
rental["City"].value_counts()


# ### There are no typos in the City column

# In[12]:


# check for typo or wrong spelling for Furnishing Status column
rental["Furnishing Status"].value_counts()


# ### There are no errors in the Furnishing Status column

# In[13]:


#check for typo or errors in the Area Type column
rental['Area Type'].value_counts()


# ### There are no errors in the Area Type Column

# ----------------------------------------------
# ### Since we are working with property data, it is impossible or does not make sense to have 0 values for certain columns
# ### we need to drop observations with value = 0 for these columns

# In[14]:


# drop BHK = 0
rental = rental[rental["BHK"] != 0].reset_index(drop=True)

# drop Rent = 0
rental = rental[rental["Rent"] != 0].reset_index(drop=True)

# drop Size = 0
rental = rental[rental["Size"] != 0].reset_index(drop=True)

# drop Bathroom = 0
rental = rental[rental["Bathroom"] != 0].reset_index(drop=True)

rental.head(10)


# In[15]:


# check the summary info of the dataframe after cleaning the data
rental.info()


# ------------------------------------------
# ## Check for Outliers
# Checking for outliers is important in preprocessing for machine learning algorithms, outliers can significantly impact the model's performance and results. Outliers can distort statistical measures, affect parameter estimation, and lead to biased predictions. By identifying and handling outliers appropriately, we can ensure more accurate and robust models that better capture the underlying patterns in the data.

# In[16]:


# take a look at the statistics of the Rent column
with pd.option_context('float_format', '{:f}'.format): print(rental["Rent"].describe())


# <b> Note:</b> The values range from 1,200 up to 3,500,000 with a median value of only 16,000 indicates that there might be presence of outliers in the dataset.

# In[17]:


# create distribution plot and boxplot to check for outliers

plt.subplot(121)
sns.distplot(rental["Rent"]);

plt.subplot(122)
rental["Rent"].plot.box(figsize=(16,5))

plt.show()


# 
# <b>Note:</b> This verifies the existence of outliers in th rent column and they will distort the models to be built later. To overcome this, we will only take into account Rent less than 100,000 to be used in model building.
# 
# 
# --------

# In[18]:


# limit the Rent to be less than or equal to 100,000
rental = rental[rental["Rent"] <= 100000].reset_index(drop=True)
print("Data type is {0} \n\nShape of dataframe is {1}\n".format(type(rental), rental.shape))


# <b>Note</b>: The data points have been reduced from <u>4734</u> to <u>4457</u> i.e. <u>277</u> outliers have been removed, which is <u>5.851%</u> of original data removed.

# In[19]:


#Check again
# take a look at the statistics of the Rent column
with pd.option_context('float_format', '{:f}'.format): print(rental["Rent"].describe())


# In[20]:


# create distribution plot and boxplot to check for outliers

plt.subplot(121)
sns.distplot(rental["Rent"]);

plt.subplot(122)
rental["Rent"].plot.box(figsize=(16,5))

plt.show()


# In[21]:


# visualize the relationship using scatter plots

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.scatter(x=rental["Size"], y=rental["Rent"])
plt.xlabel("Size")
plt.ylabel("Rent")

plt.subplot(2, 2, 2)
plt.scatter(x=rental["Bathroom"], y=rental["Rent"])
plt.xlabel("Bathroom")
plt.ylabel("Rent")

plt.subplot(2, 2, 3)
plt.scatter(x=rental["BHK"], y=rental["Rent"])
plt.xlabel("BHK")
plt.ylabel("Rent")

plt.subplot(2, 2, 4)
plt.scatter(x=rental["Floor"], y=rental["Rent"])
plt.xlabel("Floor")
plt.ylabel("Rent")

plt.show()


# In[22]:


# calculate correlation matrix
corr = rental[["Size", "Bathroom", "BHK", "Floor", "Rent"]].corr()
corr


# <b>Heat Map</b> : A heatmap is a graphical representation of data where values are encoded as colors. It is typically used to visualize the distribution and intensity of values across a two-dimensional space, such as a grid or a map.
# 
# The colors in a heatmap are used to represent the magnitude or density of the data values. Typically, a color gradient is used, where different colors correspond to different values. The specific color scheme used can vary depending on the context and the purpose of the heatmap.
# 
# In a typical heatmap color scheme, lighter or brighter colors such as yellow or white represent high values, while darker colors such as blue or black represent low values. The exact color scheme used can vary, but the general principle is that the intensity or brightness of the color corresponds to the magnitude of the value being represented.
# 

# In[23]:


# visualize correlation matrix
plt.subplots(figsize=(8,6))

fig = sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# ----------------------------------------
# ## One Hot Encoding
# 
# One-hot encoding is a popular technique used in preprocessing data for machine learning algorithms, especially when dealing with categorical variables. It transforms categorical data into a binary vector representation, allowing machine learning algorithms to effectively handle categorical features as input.
# 
# The basic idea behind one-hot encoding is to convert each categorical value into a binary vector that indicates the presence or absence of a particular category.

# In[24]:


# one hot encoding
rental_df = pd.get_dummies(rental)

rental_df.head()


# In[25]:


# check summary info to see if one hot encoding is done properly
print(rental_df.shape, "\n")
rental_df.info()


# ---------------------------------------------
# 
# # Train and Build Models
# 
# We will build several models to predict "Rent" as target variable using seven features i.e. "Size", "Bathroom", "BHK", "Floor", "City", "Furnishing Status", and "Area Type".

# In[26]:


# separate data into X features and Y target
X = rental_df.drop(columns=["Rent"])
Y = rental_df["Rent"]


# In[27]:


# split data into random train and test subsets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# --------
# # Linear Regression

# In[28]:


# fit a Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[29]:


# make predictions
y_pred = regressor.predict(X_test)


# In[30]:


# calculate R-squared
lin_r=regressor.score(X_test,y_test)
print("Linear Regression R-squared: {}".format(lin_r))


# <b>Note:</b> A linear regression R-squared value of <u>0.6752057179346523</u> signifies that approximately <u>67.52</u>% of the variance in the dependent variable (target) can be explained by the independent variables (features) in the linear regression model.

# In[31]:


#plt.scatter(y_test, y_pred)
#plt.xlabel('Actual Values')
#plt.ylabel('Predicted Values')
#plt.title('Actual vs. Predicted Values in Linear Regression')
#plt.show()
# Create a scatterplot
plt.scatter(y_test, y_pred)

# Calculate the minimum and maximum values of actual values
min_val = np.min(y_test)
max_val = np.max(y_test)

# Generate x-values for the line of best fit
x_line = np.linspace(min_val, max_val, 100)

# Calculate y-values for the line of best fit (using linear regression equation)
y_line = x_line  # Assuming a simple y = x linear relationship

# Plot the line of best fit
plt.plot(x_line, y_line, color='red')

# Add labels and title to the plot
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Scatterplot of Linear Regression: Actual Values vs. Predicted Values")

# Display the plot
plt.show()


# In[32]:


# another method of calculating R-squared
from sklearn.metrics import r2_score
lin_r2 = r2_score(y_test, y_pred)

print("Linear Regression R-squared: {}".format(lin_r2))


# In[33]:


# calculate root mean squared error (RMSE)
from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(y_pred, y_test)
lin_rmse = np.sqrt(lin_mse)
print("Linear Regression RMSE: {}".format(lin_rmse))


# In[34]:


# calculate mean absolute error (MAE)
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(y_pred, y_test)
print("Linear Regression MAE: {}".format(lin_mae))


# In[35]:


# get feature coefficients
importance = regressor.coef_
feature_indexes_by_importance = importance.argsort()
for index in feature_indexes_by_importance:
    print('{} : {}'.format(X_train.columns[index], (importance[index] )))


# In[36]:


#Plot Correlation graphs
corr_graph= sns.pairplot(rental,kind="reg",diag_kws= {'color': 'red'})

corr_graph.fig.suptitle("Correlation of House rent prediction Dataset", y=1.08)

plt.show()


# -------------------
# # Random Forest

# In[37]:


#import library
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0)


# In[38]:


# fit the model
rf.fit(X_train, y_train)


# In[39]:


# make predictions 
y_pred = rf.predict(X_test)


# In[40]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values in Random Forest')
plt.show()


# In[41]:


# calculate R-squared
forest_r=rf.score(X_test,y_test)
print("Random Forest R-squared: {}".format(forest_r))


# In[42]:


# calculate root mean squared error (RMSE)
from sklearn.metrics import mean_squared_error

forest_mse = mean_squared_error(y_pred, y_test)
forest_rmse = np.sqrt(forest_mse)
print("Random Forest Regression RMSE: {}".format(forest_rmse))


# In[43]:


# calculate mean absolute error (MAE)
from sklearn.metrics import mean_absolute_error

forest_mae = mean_absolute_error(y_pred, y_test)
print("Random Forest Regression MAE: {}".format(forest_mae))


# In[44]:


# import libraries, we will use GridSearchCV to find the best parameter values
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# In[45]:


# use gridsearch to find the best parameter

# provide range for max_depth from 1 to 5 with an interval of 1 and from 1 to 50 with an interval of 1 for n_estimators
params = {'max_depth': list(range(20, 30, 2)), 'n_estimators': list(range(30, 40, 2))}
rf = RandomForestRegressor(random_state=0)

# use gridsearch to find the best parameter
forest_reg = GridSearchCV(rf, params, cv=5)


# In[46]:


# fit the model
forest_reg.fit(X_train, y_train)


# In[47]:


# make predictions
y_pred = forest_reg.predict(X_test)


# In[48]:


# get feature importances
importance = forest_reg.best_estimator_.feature_importances_

feature_indexes_by_importance = importance.argsort()
for index in feature_indexes_by_importance:
    print('{} : {}'.format(X_train.columns[index], (importance[index] )))


# In[49]:


plt.figure(figsize=(10,6))
plt.title("Feature Importances")
feat_importances = pd.Series(forest_reg.best_estimator_.feature_importances_, index=X_train.columns)
feat_importances.sort_values().plot(kind="barh", color="Green")
# feat_importances.nlargest(20).plot(kind='barh') # top 20 features only


# ------------------------
# # Tensor Flow With Keras

# In[51]:


# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras


# ## Feature Scaling
# 
# Feature scaling, also known as data normalization, is a preprocessing technique used to standardize the range of independent variables in a dataset. By scaling the features to a common range, it ensures that variables with different scales and ranges do not dominate the learning process in a neural network model. The MinMaxScaler module from sklearn.preprocessing can be employed, setting the feature range to (0, 1).
# 
# The following steps are involved in feature scaling:
# 
# 1) Splitting the data into training and testing sets, with the possibility of including a validation set.
# 
# 2) Perform feature normalization on the training data by subtracting the mean and dividing by the variance. It is crucial to avoid incorporating future information into the training data by calculating the mean and variance of the entire dataset.
# 
# 3) Apply the same normalization parameters learned from the training data to normalize the testing instances. This ensures evaluation on unseen data and the ability to assess the model's generalization capabilities.
# 
# To implement feature scaling, the fit_transform() function is used on the training data to learn the scaling parameters and simultaneously scale the data. However, only the transform() function is applied to the testing data, using the learned scaling parameters from the training data.

# In[52]:


print(X_train.shape)
X_train.head()


# In[53]:


print(X_test.shape)
X_test.head()


# In[54]:


# import MinMaxScaler from sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler

# define the scaler 
scaler = MinMaxScaler(feature_range=(0, 1))

X_train[["Size", "Bathroom", "BHK", "Floor"]] = scaler.fit_transform(X_train[["Size", "Bathroom", "BHK", "Floor"]])

X_test[["Size", "Bathroom", "BHK", "Floor"]] = scaler.transform(X_test[["Size", "Bathroom", "BHK", "Floor"]])


# In[55]:


print(X_train.shape)
X_train.head()


# In[56]:


print(X_test.shape)
X_test.head()


# ## Create the model
# 
# We will utilize the Keras Sequential model to construct our neural network. The Sequential model allows for a linear stack of layers, making it simple to create the model by passing a list of layer instances to the constructor, which is set up using the command model = Sequential().
# 
# For this model we selecte two densely connected hidden layers with 64 hidden units each. This choice determines the degree of flexibility the network has in learning representations. Using more hidden units enables the network to learn more complex representations, but it also increases computational cost and the risk of overfitting.
# 
# Overfitting occurs when the model becomes too complex, capturing random noise rather than the underlying relationship. To mitigate overfitting, especially when training data is limited, it is advisable to employ a smaller network with fewer hidden layers.
# 
# Therfore, we will use the Rectified Linear Unit (ReLU) activation function, which is widely used due to its effectiveness. However, feel free to experiment with other activation functions like Hyperbolic Tangent (tanh).
# 
# To gain insights into the model, we can use attributes such as output_shape or employ the summary() function. Additionally, the get_config() function allows us to retrieve the configuration information of the model.

# In[57]:


# import Sequential from keras.models
from keras.models import Sequential

# import Dense from keras.layers
from keras.layers import Dense

# initialize the constructor
model = Sequential()

# add a densely-connected layer with 64 units to the model
model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))

# add another layer with 64 units
model.add(Dense(64, activation="relu"))

# add an output layer with 1 output unit
model.add(Dense(1))


# In[58]:


# model output shape
print(model.output_shape)

# model summary
print(model.summary())


# In[59]:


# model config
model.get_config()


# ## Set up training
# 
# After constructing the model, the next step is to configure its learning process by using the compile method. The compile method accepts three crucial arguments:
# 
# 1. Optimizer: This argument specifies the training procedure by providing an optimizer instance from the tf.train module. Examples of optimizers include tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, or tf.train.GradientDescentOptimizer. For this case, we will use tf.train.RMSPropOptimizer with a learning rate of 0.001.
# 
# 2. Loss: The loss function to minimize during the optimization process. Common choices include mean square error (mse), categorical_crossentropy, or binary_crossentropy. The loss function can be specified either by name or by passing a callable object from the tf.keras.losses module. In this scenario, we will utilize the mean square error (mse) loss.
# 
# 3. Metrics: These are used to monitor the training progress. They can be either string names or callable objects from the tf.keras.metrics module. In this case, we will use mean absolute error (mae) as the metric to track during training.

# In[60]:


optimizer=tf.keras.optimizers.RMSprop(0.001)
model.compile(loss="mse",
              optimizer=optimizer,
              metrics=["mae"])


# ## Train the Model
# 
# The model will be trained for 50 epochs, which means it will go through the entire training dataset 50 times. During training, 20% of the training data will be set aside as validation data. The model will not be trained on this validation data but will instead evaluate the loss and any model metrics on this data at the end of each epoch.
# 
# The batch_size will be set to the default value of 32. This means that during each training iteration, the model will process 32 samples at a time before updating the weights.
# 
# To monitor the training progress, the verbose argument is set to 1. This will display logs with an animated progress bar, showing the training progress for each epoch. If verbose is set to 0, nothing will be displayed (silent), while verbose=2 will only mention the number of epochs completed without showing the progress bar.

# In[61]:


#Train the Model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)


# In[62]:


# list all data in history
print(history.history.keys())


# In[63]:


# summarize history for mean_absolute_error
plt.figure(figsize=(10,6))
plt.plot(history.history["mae"])
plt.plot(history.history["val_mae"])
plt.title("Mean Absolute Error")
plt.ylabel("MAE")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"])
plt.show()


# In[64]:


# summarize history for loss
plt.figure(figsize=(10,6))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"])
plt.show()


# ## Predict Values

# In[65]:


#Predict values
y_pred = model.predict(X_test)


# In[66]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values TF Keras')
plt.show()


# ## Evaluate Model

# In[67]:


#Evaluate Model
[mse, mae] = model.evaluate(X_test, y_test, verbose=1)


# In[68]:


#Calculate RMSE and MAE
keras_rmse = np.sqrt(int(mse))
keras_mae = mae

print("Testing set Root Mean Squared Error: {}".format(keras_rmse))
print("Testing set Mean Absolute Error: {}".format(keras_mae))


# In[69]:


# calculate R-squared
from sklearn.metrics import r2_score
keras_r2 = r2_score(y_test, y_pred)

print("Tensorflow with Keras Sequential model R-squared: {}".format(keras_r2))


# In[70]:


# Set the labels for the x-axis
labels = ['Linear Regression', 'Random Forest', 'TF Keras']

# Set the values for R-squared, RMSE, and MAE
r_values = [lin_r, forest_r, keras_r2]
rmse_values = [lin_rmse, forest_rmse, keras_rmse]
mae_values = [lin_mae, forest_mae, keras_mae]

# Set the positions of the bars on the x-axis
x = np.arange(len(labels))

# Set the width of the bars
width = 0.4

# Set the color scheme for each model
colors = ['#001F3F', '#FF851B', '#800000']  # Navy blue, dark orange, maroon

# Create the figure and axes
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

# Plot the R-squared values
ax1.bar(x, r_values, width, color=colors)
ax1.set_ylabel('R-squared')
ax1.set_ylim(0, 1)

# Add values on top of bars
for rect, r, color in zip(ax1.patches, r_values, colors):
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width() / 2, height, f'{r:.2f}',
             ha='center', va='bottom', color='black')

# Plot the RMSE values
ax2.bar(x, rmse_values, width, color=colors)
ax2.set_ylabel('RMSE')

# Add values on top of bars
for rect, rmse, color in zip(ax2.patches, rmse_values, colors):
    height = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width() / 2, height, f'{rmse:.2f}',
             ha='center', va='bottom', color='black')

# Plot the MAE values
ax3.bar(x, mae_values, width, color=colors)
ax3.set_xlabel('Models')
ax3.set_ylabel('MAE')

# Add values on top of bars
for rect, mae, color in zip(ax3.patches, mae_values, colors):
    height = rect.get_height()
    ax3.text(rect.get_x() + rect.get_width() / 2, height, f'{mae:.2f}',
             ha='center', va='bottom', color='black')

# Set the x-axis tick labels and title
plt.xticks(x, labels)
plt.suptitle('Comparison: Linear Regression vs. Random Forest')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# In[71]:


# combine all models' results into one dataframe
data = {"Model": ["Linear Regression", "Random Forest","TF Keras"], 
        "R-squared": [lin_r, forest_r,keras_r2],           
        "RMSE": [lin_rmse, forest_rmse,keras_rmse],
        "MAE": [lin_mae, forest_mae,keras_mae]}

results = pd.DataFrame(data=data)
results

