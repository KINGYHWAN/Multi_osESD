
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import data_config

# def run_KNN(x_data, y_data, params, train_percent):
#     # values = x_data.values.reshape(-1, 1)
#     values = x_data.iloc[:int(len(x_data)*train_percent)].values
#     y_train = y_data.iloc[:int(len(x_data)*train_percent)]
#     anom_percent = sum(y_train)/len(y_train)
#     if anom_percent==0:
#         anom_percent = 0.01
#     else:
#         anom_percent *= 0.75

#     # params.neighbors = 5
#     # knn_model = NearestNeighbors(n_neighbors=params.neighbors)

#     knn_model = NearestNeighbors()
#     pipeline = Pipeline([
#     ('scaler', StandardScaler()),  # Feature scaling
#     ('knn', knn_model)
#     ])

#     param_grid = {
#         'knn__n_neighbors': [3, 5, 7, 10],  # Different values for k
#         'knn__metric': ['euclidean', 'manhattan', 'minkowski']  # Different distance metrics
#     }
#     # knn_model.fit(x_data)
#     grid_search = GridSearchCV(pipeline, param_grid, cv=5,verbose=1)
#     grid_search.fit(x_data, y_train)

#     best_model = grid_search.best_estimator_
#     best_model.fit(x_data, y_train) 

#     distances, indices = knn_model.kneighbors(x_data)
#     anomaly_scores = np.mean(distances, axis=1)
#     # print(anom_percent)
#     threshold = np.percentile(anomaly_scores, 1-anom_percent)  
#     anom_preds = np.where(anomaly_scores > threshold, 1, 0)

#     if train_percent<1:
#         test_preds = anom_preds[int(len(x_data)*train_percent):]
#     else:
#         test_preds = anom_preds[train_percent:]
    
#     return test_preds


from sklearn.neighbors import KNeighborsClassifier

def run_KNN(train_df, test_df, params):
    # Split data into training and testing sets
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_percent, random_state=42)

    # Define the k-NN classifier within a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Feature scaling
        ('knn', KNeighborsClassifier())
    ])

    # Grid of parameters to search
    param_grid = {
        'knn__n_neighbors': [params.neighbors],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
        'knn__weights': ['uniform', 'distance']  # Additional parameter for tuning
    }

    # Grid search to find the best hyperparameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', verbose=1)
    grid_search.fit(train_df[0], train_df[1])  # Fitting only on training data

    # Retrieve the best model and evaluate
    best_model = grid_search.best_estimator_
    test_preds = best_model.predict(test_df[0])  # Prediction on test data
    # test_score = best_model.score(test_preds, test_df[1].reshape(1,-1))  # Scoring on test data
    return test_preds.reshape((len(test_preds), 1))
    # return test_preds, test_score, y_test


# def run_KNN(train_df, test_df, params):
#     # Split data into features and target
#     x_train, y_train = train_df[0], train_df[1]
#     x_test, y_test = test_df[0], test_df[1]

#     # Define the k-NN classifier within a pipeline using provided parameters
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),  # Feature scaling for better performance
#         ('knn', KNeighborsClassifier(
#             n_neighbors=params['n_neighbors'],
#             metric=params['metric'],
#             weights=params['weights']))
#     ])

#     # Fit the model on the training data
#     pipeline.fit(x_train, y_train)

#     # Predict using the test data
#     test_preds = pipeline.predict(x_test)

#     # Optionally calculate and return the F1 score along with predictions
#     test_score = f1_score(y_test, test_preds, average='weighted')

#     return test_preds, test_score
