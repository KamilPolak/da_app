# Import necessary libraries
import json

from matplotlib import pyplot as plt
import joblib

import pandas as pd
import streamlit as st
import numpy as np

# Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report

# Custom classes 
from .utils import isNumerical
import os

def app():
    """This application helps in running machine learning models without having to write explicit code 
    by the user. It runs some basic models and let's the user select the X and y variables. 
    """
    
    # Load the data 
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        data = pd.read_csv('data/main_data.csv')

        # Create the model parameters dictionary 
        params = {}

        # Use two column technique 
        col1, col2 = st.beta_columns(2)

        # Design column 1 
        y_var = col1.radio("Select the variable to be predicted (y)", options=data.columns)

        # Design column 2 
        X_var = col2.multiselect("Select the variables to be used for prediction (X)", options=data.columns)

        # Check if len of x is not zero 
        if len(X_var) == 0:
            st.error("You have to put in some X variable and it cannot be left empty.")

        # Check if y not in X 
        if y_var in X_var:
            st.error("Warning! Y variable cannot be present in your X-variable.")

        # Option to select predition type 
        pred_type = st.radio("Select the type of process you want to run.", 
                            options=["Regression", "Classification"], 
                            help="Write about reg and classification")

        # Add to model parameters 
        params = {
                'X': X_var,
                'y': y_var, 
                'pred_type': pred_type,
        }

        # if st.button("Run Models"):

        st.write(f"**Variable to be predicted:** {y_var}")
        st.write(f"**Variable to be used for prediction:** {X_var}")
        
        # Divide the data into test and train set 
        X = data[X_var]
        y = data[y_var]

        # Perform data imputation 
        # st.write("THIS IS WHERE DATA IMPUTATION WILL HAPPEN")
        
        # Perform encoding
        X = pd.get_dummies(X)

        # Check if y needs to be encoded
        if not isNumerical(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            
            # Print all the classes 
            st.write("The classes and the class allotted to them is the following:-")
            classes = list(le.classes_)
            for i in range(len(classes)):
                st.write(f"{classes[i]} --> {i}")
        

        # Perform train test splits 
        st.markdown("#### Train Test Splitting")
        size = st.slider("Percentage of value division",
                            min_value=0.1, 
                            max_value=0.9, 
                            step = 0.1, 
                            value=0.8, 
                            help="This is the value which will be used to divide the data for training and testing. Default = 80%")

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
        st.write("Number of training samples:", X_train.shape[0])
        st.write("Number of testing samples:", X_test.shape[0])

        # Save the model params as a json file
        with open('data/metadata/model_params.json', 'w') as json_file:
            json.dump(params, json_file)

        ''' RUNNING THE MACHINE LEARNING MODELS '''
        if pred_type == "Regression":
            st.write("Running Regression Models on Sample")

            # Table to store model and accurcy 
            model_r2 = []

            # Linear regression model 
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_r2 = lr_model.score(X_test, y_test)
            model_r2.append(['Linear Regression', lr_r2])

            # Decision Tree model 
            dt_model = DecisionTreeRegressor()
            dt_model.fit(X_train, y_train)
            dt_r2 = dt_model.score(X_test, y_test)
            model_r2.append(['Decision Tree Regression', dt_r2])

            # Save one of the models 
            if dt_r2 > lr_r2:
                # save decision tree 
                joblib.dump(dt_model, 'data/metadata/model_reg.sav')
            else: 
                joblib.dump(lr_model, 'data/metadata/model_reg.sav')

            # Make a dataframe of results 
            results = pd.DataFrame(model_r2, columns=['Models', 'R2 Score']).sort_values(by='R2 Score', ascending=False)
            st.dataframe(results)
        
        if pred_type == "Classification":
            st.write("Running Classfication Models on Sample")

            # Table to store model and accurcy
            # Not used - replaced by classification report 
            # model_acc = []

            # Linear regression model 
            lc_model = LogisticRegression()
            lc_model.fit(X_train, y_train)
            lc_acc = lc_model.score(X_test, y_test)
            #model_acc.append(['Logistic Regression', lc_acc])
            y_pred = lc_model.predict(X_test)
            st.text('Logistic Regression Report:\n ' + classification_report(y_test, y_pred))

            #Learning curve - Logistic Regression
            train_sizes, train_scores, test_scores = learning_curve(LogisticRegression(), X_train, y_train, n_jobs=-1, cv=10, train_sizes=np.linspace(.1, 1.0, 10), verbose=0)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            fig = plt.figure()
            plt.title("Learning Curve")
            plt.legend(loc="best")
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            plt.gca().invert_yaxis()
            
            # box-like grid
            plt.grid()
            
            # plot the std deviation as a transparent range at each training set size
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
            
            # plot the average training and test score lines at each training set size
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            
            # sizes the window for readability and displays the plot
            # shows error from 0 to 1.1
            plt.ylim(-.1,1.1)
            st.pyplot(fig)

            # Decision Tree model 
            dtc_model = DecisionTreeClassifier()
            dtc_model.fit(X_train, y_train)
            dtc_acc = dtc_model.score(X_test, y_test)
            #model_acc.append(['Decision Tree Regression', dtc_acc])
            y_pred1 = dtc_model.predict(X_test)
            st.text('Decision Tree Report:\n ' + classification_report(y_test, y_pred1))

            # Save one of the models 
            if dtc_acc > lc_acc:
                # save decision tree 
                joblib.dump(dtc_model, 'data/metadata/model_classification.sav')
            else: 
                joblib.dump(lc_model, 'data/metadata/model_classificaton.sav')

            # Make a dataframe of results 
            # not used - replaced by classification report
            #results = pd.DataFrame(model_acc, columns=['Models', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
            #st.dataframe(results)
            

            #Learning curve - Decision Tree
            train_sizes, train_scores, test_scores = learning_curve(DecisionTreeClassifier(), X_train, y_train, n_jobs=-1, cv=10, train_sizes=np.linspace(.1, 1.0, 10), verbose=0)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            fig = plt.figure()
            plt.title("Learning Curve")
            plt.legend(loc="best")
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            plt.gca().invert_yaxis()
            
            # box-like grid
            plt.grid()
            
            # plot the std deviation as a transparent range at each training set size
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
            
            # plot the average training and test score lines at each training set size
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            
            # sizes the window for readability and displays the plot
            # shows error from 0 to 1.1
            plt.ylim(-.1,1.1)
            st.pyplot(fig)