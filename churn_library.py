# library doc string
'''
Module contains functions of churn customer analysis

Author : Ifegwu Daniel Agbanyim

Date : 26th October 2022
'''
# Import libraries
import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import table
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, plot_roc_curve
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(file_path):
    '''
    returns dataframe for the csv found at pth

    input:
            file_path: a path to the csv
    output:
            df: pandas dataframe
    '''

    dataframe = pd.read_csv(file_path)  # read the csv file
    return dataframe  # output the data frame


def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    # copy dataframe
    eda_df = dataframe.copy(deep=True)

    # DataFrame
    figure, axis = plt.subplots(figsize=(12, 2))  # set size frame
    axis.xaxis.set_visible(False)  # hide the x axis
    axis.yaxis.set_visible(False)  # hide the y axis
    axis.set_frame_on(False)  # no visible frame, uncomment if size is ok
    tabla = table(axis, eda_df.head(10), loc='upper left', colWidths=[
                  0.12] * len(eda_df.columns))  # where eda_df is your data frame
    tabla.auto_set_font_size(False)  # Activate set fontsize manually
    tabla.set_fontsize(9)  # if ++fontsize is necessary ++colWidths
    tabla.scale(1.2, 1.2)  # change size table
    plt.savefig(fname='./images/eda/dataframe.png')
    plt.close()

    # Churn
    eda_df['Churn'] = eda_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # churn distributions
    plt.figure(figsize=(20, 10))
    eda_df['Churn'].hist()
    plt.savefig(fname='./images/eda/churn_dist.png')
    plt.close()

    # Customer Age Distribution
    plt.figure(figsize=(20, 10))
    eda_df['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/customer_age_dist.png')
    plt.close()

    # Marital Status Distribution
    plt.figure(figsize=(20, 10))
    eda_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/marital_status_dist.png')
    plt.close()

    # Total Transaction Distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(eda_df['Total_Trans_Ct'], kde=True)
    plt.savefig(fname='./images/eda/total_transaction_dist.png')
    plt.close()

    # Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(eda_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/heatmap.png')
    plt.close()

    return eda_df


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
                        be used for naming variables or index y column]

    output:
        data_frame: pandas dataframe with new columns for
    '''
    # Copy DataFrmae
    encoder_df = dataframe.copy(deep=True)

    # for category in category_lst:
    #     column_lst = []
    #     column_groups = dataframe.groupby(category).mean()['Churn']

    #     for val in dataframe[category]:
    #         column_lst.append(column_groups.loc[val])

    #     if response:
    #         encoder_df[category + '_' + response] = column_lst
    #     else:
    #         encoder_df[category] = column_lst

    # test run `Column names should be thesame` could not pass
    # for category in category_lst:
    #     column_lst = category
    #     encoder_df[column_lst] = dataframe.groupby(
    #         category)["Churn"].transform("mean")

    for category in category_lst:
        column_lst = category
        if response:
            column_lst = category + '_' + response
            encoder_df[column_lst] = dataframe.groupby(
                category)["Churn"].transform("mean")
        else:
            encoder_df[category] = column_lst

    return encoder_df


def perform_feature_engineering(dataframe, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could
                            be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # categorical features
    cat_columns = ['Gender', 'Education_Level',
                   'Marital_Status', 'Income_Category', 'Card_Category']

    # feature engineering
    encode_df = encoder_helper(
        dataframe=dataframe, category_lst=cat_columns, response=response)

    # target feature
    y = encode_df['Churn']

    # Create dataframe
    X = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    # Apply dataframe
    X[keep_cols] = encode_df[keep_cols]

    # Train and Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return (X_train, X_test, y_train, y_test)


def classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf):
    '''
    produces classification report for training and testing
    results and stores report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Random Forest Classifier
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/rf_results.png')
    plt.close()

    # Logistic Regression Classifier
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_results.png')
    plt.close()


def feature_importance_plot(model, features, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values output_pth: path to store the figure

    output:
             None
    '''
    # Feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Sorted feature importances
    names = [features.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(25, 25))

    # Create plot title
    plt.title('Feature Importance')
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(features.shape[1]), importances[indices])

    # x-axis labels
    plt.xticks(range(features.shape[1]), names, rotation=90)

    # Save image
    plt.savefig(fname=output_pth + 'feature_importances.png')
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Random Forest Classifier and Logistic Regression
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
    lrc = LogisticRegression(n_jobs=-1, max_iter=1000)

    # Parameters for Grid Search
    param_grid = {
        'n_estimators': [
            200, 500], 'max_features': [
            'auto', 'sqrt'], 'max_depth': [
                4, 5, 100], 'criterion': [
                    'gini', 'entropy']}

    # Grid searvh and fit for Random Forest Classifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Logistic Regression
    lrc.fit(X_train, y_train)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Compute train and test predictions for Random Forest Classifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Compute train and test predictions for Logistic Regression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Computer ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_,
                              X_test, y_test, ax=axis, alpha=0.8)
    lrc_plot = plot_roc_curve(lrc, X_test, y_test, ax=axis, alpha=0.8)
    plt.savefig(fname='./images/results/roc_curve_result.png')
    plt.close()

    classification_report_image(
        y_train, y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf
    )

    # Compute and feature importance
    feature_importance_plot(model=cv_rfc, features=X_test,
                            output_pth='./images/results/')


if __name__ == "__main__":
    # Import data
    BANK_DF = import_data(file_path='./data/bank_data.csv')

    # perform EDA
    EDA_DF = perform_eda(dataframe=BANK_DF)

    # Performance Feature Engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        dataframe=EDA_DF, response='Churn')

    # Model taining, prediction and evaluation
    train_models(X_train=X_TRAIN, X_test=X_TEST,
                 y_train=Y_TRAIN, y_test=Y_TEST)
