#!/usr/bin/env python
import pandas as pd
from pathlib import Path
import os
import numpy as np
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.metrics import confusion_matrix

from operator import itemgetter
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

from signal_producer import signals_generator
from feature_generate import feature_generate, features_selection
from visualization import plotting_signal

# In[2]:


pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD']

# In[16]:


Dataset_dir = 'ohlc_data'
root_path = Path(os.getcwd())
for i in pairs:
    dataset_path = os.path.join(root_path, Dataset_dir, i, 'H1_' + i + '_2015' + ".csv")
    df = pd.read_csv(dataset_path)
    print(dataset_path)

    if 'volume' not in df.columns:
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'},
                  inplace=True)
    else:
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
                  inplace=True)

    #     print(df)
    f_df = feature_generate(df)
    processed = f_df.dropna().reset_index()

    if i != 'USDJPY':
        #     print(f_df)
        signals, stats = signals_generator(processed.SMA_10.values, 30, 0.002932)
        print(stats)
        #         plotting_signal(processed.SMA_10.values, signals)

        processed['labels'] = pd.Series(signals)
        processed['labels'] = processed['labels'].astype(np.int8)
        #         processed = processed.drop(columns=['index'])
        # Usage with topk='all' to select all available features after initial reduction
        selected_ftr, f_idx = features_selection(processed, selection_method='all', topk='all', num_features=16)

        # Extract selected features based on indices and ensure only numeric data is selected
        X = processed.iloc[:, f_idx].select_dtypes(include=[np.number]).values
        y = processed['labels'].values  # Target variable

        # Print total selected features
        print('Total number of selected features:', len(selected_ftr))

        # Splitting the data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            X, y,
            train_size=0.8,
            test_size=0.2,
            random_state=2,
            shuffle=True,
            stratify=y
        )

        # Uncomment the following lines if you want to handle class imbalance
        smote = RandomOverSampler(random_state=42, sampling_strategy='not majority')
        x_train, y_train = smote.fit_resample(x_train, y_train)
        print('Resampled dataset shape:', Counter(y_train))

        # Dynamic train-validation split
        train_split = 0.8 if 0.7 * x_train.shape[0] < 2500 else 0.7
        print('train_split =', train_split)

        # Split the training set further into training and cross-validation sets
        x_train, x_cv, y_train, y_cv = train_test_split(
            x_train, y_train,
            train_size=train_split,
            test_size=1 - train_split,
            random_state=2,
            shuffle=True,
            stratify=y_train
        )

        # Choose between MinMaxScaler or StandardScaler
        scaler = MinMaxScaler(feature_range=(0, 1))  # For Min-Max scaling
        # scaler = StandardScaler()                   # For Standard scaling

        # Apply scaling to training, cross-validation, and test sets
        x_train = scaler.fit_transform(x_train)
        x_cv = scaler.transform(x_cv)
        x_test = scaler.transform(x_test)

        # Copying the main training set for further operations if needed
        x_main = x_train.copy()

        # Printing the shapes of train, cross-validation, and test sets
        print("Shape of x, y train/cv/test:",
              x_train.shape, y_train.shape, x_cv.shape, y_cv.shape, x_test.shape, y_test.shape)

        # Ensure reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        # Reshape inputs for CNN
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))  # Add channel dimension
        x_cv = x_cv.reshape((x_cv.shape[0], x_cv.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        # Normalize data if not already normalized
        x_train = x_train / np.max(x_train)
        x_cv = x_cv / np.max(x_cv)
        x_test = x_test / np.max(x_test)

        #         pass
        #     else:
        #         #     print(f_df)
        #         signals, stats = signals_generator(processed.SMA_10.values, 30, 2)
        #         print('USDJPY: ', stats)
        #         plotting_signal(processed.SMA_10.values, signals)

        # Assuming `y_train`, `y_cv`, and `y_test` are integer labels
        y_train_onehot = to_categorical(y_train, num_classes=3)
        y_cv_onehot = to_categorical(y_cv, num_classes=3)
        y_test_onehot = to_categorical(y_test, num_classes=3)

        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(15, 1)),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')  # 3 classes: 0, 1, 2
        ])

        ## If One hot encoded
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        ### If one hot encoded
        history = model.fit(x_train, y_train_onehot,
                            epochs=50,
                            batch_size=32,
                            validation_data=(x_cv, y_cv_onehot))

        # if One hot the Evaluate the model like below
        test_loss, test_accuracy = model.evaluate(x_test, y_test_onehot)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Get predictions
        y_pred = model.predict(x_test)  # This returns probabilities
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices

        # Convert one-hot encoded y_test back to class indices for comparison
        y_test_classes = np.argmax(y_test_onehot, axis=1)

        # Check some predictions
        print("Predicted:", y_pred_classes[:10])
        print("Actual:", y_test_classes[:10])

        conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
        print(conf_matrix)
        print(classification_report(y_test_classes, y_pred_classes))

    break




df_processed = indicators_df.dropna().reset_index()

predicted_prices = df_processed.SMA_10.values


# In[8]:





# In[6]:





# In[17]:


signals, stats = signals_generator(df_processed.SMA_10.values, 30, 3)
stats

# In[18]:


plotting_signal(df_processed.SMA_10.values, signals)

# In[19]:


print(len(df_processed.SMA_10.values), len(df_processed), len(signals))

# In[20]:


df_processed['labels'] = pd.Series(signals)
df_processed['labels'] = df_processed['labels'].astype(np.int8)
df_processed = df_processed.drop(columns=['index'])

# In[21]:


df_processed

# In[22]:


df_processed.columns


# ## Explanation of the Script
# #### Correlation Matrix: Computes the correlation matrix and removes features that are highly correlated (above a threshold, e.g., 0.85).
# #### Lasso: Uses Lasso regression with cross-validation to select features. Lasso shrinks less important feature coefficients to zero.
# #### RFE with Random Forest: Performs Recursive Feature Elimination (RFE) with Random Forest to rank and select the top 10 features.
# #### Random Forest Importance: Trains a Random Forest and plots the top 10 features based on importance scores.
# #### Final Selection: Combines the selected features from Lasso and RFE (can use intersection or union) and creates a reduced dataset X_selected for further modeling.

# In[23]:


def features_selection(df):
    # Separate features and target variable
    X = df.drop(columns=['labels', 'timestamp'])  # Drop 'timestamp' and target column 'labels'
    y = df['labels']  # Target

    # Ensure that only numeric columns are used for correlation
    X_numeric = X.select_dtypes(include=[np.number])

    # Step 1: Correlation Matrix
    # Remove highly correlated features
    cor_matrix = X_numeric.corr().abs()
    upper_triangle = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.85)]
    X.drop(columns=high_corr_features, inplace=True)
    print(f"Features removed due to high correlation: {high_corr_features}")

    # Step 2: Feature Selection with Lasso
    # Scale the data before Lasso
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply LassoCV for feature selection
    lasso = LassoCV(cv=5, random_state=0).fit(X_scaled, y)
    lasso_selected_features = X.columns[lasso.coef_ != 0]
    print(f"Features selected by Lasso: {list(lasso_selected_features)}")

    # Step 3: Recursive Feature Elimination (RFE) with Random Forest
    rfe_selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=10, step=1)
    rfe_selector = rfe_selector.fit(X, y)
    rfe_selected_features = X.columns[rfe_selector.support_]
    print(f"Features selected by RFE: {list(rfe_selected_features)}")

    # Step 4: Feature Importance using Random Forest
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X, y)

    # Combine all selected features
    selected_features = list(set(lasso_selected_features) | set(rfe_selected_features))
    print(f"Final selected features: {selected_features}")

    # Define your selection method and topk (number of top features to select)
    selection_method = 'all'  # Options: 'anova', 'mutual_info', 'all'
    topk = 10  # Adjust this number based on your preference for top features

    if selection_method == 'anova' or selection_method == 'all':
        select_k_best_anova = SelectKBest(f_classif, k=topk)
        select_k_best_anova.fit(X, y)

        # Get the selected features by ANOVA
        selected_features_anova = itemgetter(*select_k_best_anova.get_support(indices=True))(list_features)
        print("Selected features by ANOVA:", selected_features_anova)
        print("Indices of selected features by ANOVA:", select_k_best_anova.get_support(indices=True))
        print("****************************************")

    if selection_method == 'mutual_info' or selection_method == 'all':
        select_k_best_mic = SelectKBest(mutual_info_classif, k=topk)
        select_k_best_mic.fit(X, y)

        # Get the selected features by Mutual Information
        selected_features_mic = itemgetter(*select_k_best_mic.get_support(indices=True))(list_features)
        print("Selected features by Mutual Information:", selected_features_mic)
        print("Indices of selected features by Mutual Information:", select_k_best_mic.get_support(indices=True))

    return selected_features


# In[14]:




# In[11]:


# Usage with topk='all' to select all available features after initial reduction
selected_ftr, f_idx = features_selection(df_processed, selection_method='all', topk='all', num_features=16)

# In[26]:


df_processed['labels'].value_counts()

# In[27]:


# # Separate the features and target again from df_processed
# X_selected = df_processed[selected_ftr]  # Use only selected features
# y = df_processed['labels']  # Define the target variable

# # Split the data for model training
# X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)


# In[28]:


# Extract selected features based on indices and ensure only numeric data is selected
X = df_processed.iloc[:, f_idx].select_dtypes(include=[np.number]).values
y = df_processed['labels'].values  # Target variable

# Print total selected features
print('Total number of selected features:', len(selected_ftr))

# Splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.8,
    test_size=0.2,
    random_state=2,
    shuffle=True,
    stratify=y
)

# Uncomment the following lines if you want to handle class imbalance
smote = RandomOverSampler(random_state=42, sampling_strategy='not majority')
x_train, y_train = smote.fit_resample(x_train, y_train)
print('Resampled dataset shape:', Counter(y_train))

# Dynamic train-validation split
train_split = 0.8 if 0.7 * x_train.shape[0] < 2500 else 0.7
print('train_split =', train_split)

# Split the training set further into training and cross-validation sets
x_train, x_cv, y_train, y_cv = train_test_split(
    x_train, y_train,
    train_size=train_split,
    test_size=1 - train_split,
    random_state=2,
    shuffle=True,
    stratify=y_train
)

# Choose between MinMaxScaler or StandardScaler
scaler = MinMaxScaler(feature_range=(0, 1))  # For Min-Max scaling
# scaler = StandardScaler()                   # For Standard scaling

# Apply scaling to training, cross-validation, and test sets
x_train = scaler.fit_transform(x_train)
x_cv = scaler.transform(x_cv)
x_test = scaler.transform(x_test)

# Copying the main training set for further operations if needed
x_main = x_train.copy()

# Printing the shapes of train, cross-validation, and test sets
print("Shape of x, y train/cv/test:",
      x_train.shape, y_train.shape, x_cv.shape, y_cv.shape, x_test.shape, y_test.shape)

# In[48]:


_labels, _counts = np.unique(y_train, return_counts=True)
print("percentage of class 0 = {}, class 1 = {}, class 2 = {}".format(_counts[0] / len(y_train) * 100,
                                                                      _counts[1] / len(y_train),
                                                                      _counts[2] / len(y_train)
                                                                      )
      )

# ### Starting of Deep Learning CNN based classification part

# In[30]:


# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# #### Prepare Data Assuming you have the data ready as
# #### x_train, y_train, x_cv, y_cv, x_test, y_test. 
# #### CNN expects data with an additional dimension for channels.

# In[31]:


# Reshape inputs for CNN
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))  # Add channel dimension
x_cv = x_cv.reshape((x_cv.shape[0], x_cv.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Normalize data if not already normalized
x_train = x_train / np.max(x_train)
x_cv = x_cv / np.max(x_cv)
x_test = x_test / np.max(x_test)

# ### If want to Use One-Hot Encoding
# ### Convert Labels to One-Hot:

# In[32]:


# Assuming `y_train`, `y_cv`, and `y_test` are integer labels
y_train_onehot = to_categorical(y_train, num_classes=3)
y_cv_onehot = to_categorical(y_cv, num_classes=3)
y_test_onehot = to_categorical(y_test, num_classes=3)

# In[ ]:


# ### Build the CNN Model

# In[33]:


model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(15, 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: 0, 1, 2
])

# ### Compile Model and Train afterwards

# In[34]:


# model.compile(optimizer='adam', 
#               loss='sparse_categorical_crossentropy', 
#               metrics=['accuracy'])


# In[35]:


## If One hot encoded
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# In[36]:


# history = model.fit(x_train, y_train, 
#                     epochs=50, 
#                     batch_size=32, 
#                     validation_data=(x_cv, y_cv))


# In[ ]:


# In[37]:


### If one hot encoded
history = model.fit(x_train, y_train_onehot,
                    epochs=50,
                    batch_size=32,
                    validation_data=(x_cv, y_cv_onehot))

# ### Evaluate the Model

# In[38]:


# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print(f"Test Accuracy: {test_accuracy:.2f}")


# In[39]:


# if One hot the Evaluate the model like below
test_loss, test_accuracy = model.evaluate(x_test, y_test_onehot)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# In[40]:


# Get predictions
y_pred = model.predict(x_test)  # This returns probabilities
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices

# Convert one-hot encoded y_test back to class indices for comparison
y_test_classes = np.argmax(y_test_onehot, axis=1)

# Check some predictions
print("Predicted:", y_pred_classes[:10])
print("Actual:", y_test_classes[:10])

# In[41]:


conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
print(conf_matrix)
print(classification_report(y_test_classes, y_pred_classes))

# ### Visualize Training

# In[42]:


# import matplotlib.pyplot as plt

# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


# In[43]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ### Predictions

# In[44]:


# y_pred = model.predict(x_test)
# y_pred_classes = np.argmax(y_pred, axis=1)

# # Compare predictions
# print(classification_report(y_test, y_pred_classes))


# ### Compute Confusion Matrix

# In[45]:


# Assuming `y_test` is true labels and `y_pred_classes` are predicted labels
# conf_matrix = confusion_matrix(y_test, y_pred_classes)
# print(conf_matrix)


# In[46]:


# # Create a heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()


# In[47]:


# one hot encoded Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# In[ ]:
