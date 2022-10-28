# library doc string
'''
Module contains functions of churn customer analysis

Author : Ifegwu Daniel Agbanyim

Date : 26th October 2022
'''
# Import libraries
import pandas as pd
import seaborn as sns
from pandas.plotting import table
import matplotlib.pyplot as plt
import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(file_path):
    '''
    returns dataframe for the csv found at pth

    input:
            file_path: a path to the csv
    output:
            df: pandas dataframe
    '''

    df = pd.read_csv(file_path)  # read the csv file
    return df  # output the data frame


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
    fig, ax = plt.subplots(figsize=(12, 2))  # set size frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
    tabla = table(ax, eda_df.head(10), loc='upper left', colWidths=[
                  0.12]*len(eda_df.columns))  # where eda_df is your data frame
    tabla.auto_set_font_size(False)  # Activate set fontsize manually
    tabla.set_fontsize(9)  # if ++fontsize is necessary ++colWidths
    tabla.scale(1.2, 1.2)  # change size table
    plt.savefig(fname='./images/eda/dataframe.png')

    # Churn
    eda_df['Churn'] = eda_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # churn distributions
    plt.figure(figsize=(20, 10))
    eda_df['Churn'].hist()
    plt.savefig(fname='./images/eda/churn_dist.png')

    # Customer Age Distribution
    plt.figure(figsize=(20, 10))
    eda_df['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/customer_age_dist.png')

    # Marital Status Distribution
    plt.figure(figsize=(20, 10))
    eda_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/marital_status_dist.png')

    # Total Transaction Distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(eda_df['Total_Trans_Ct'], kde=True)
    plt.savefig(fname='./images/eda/total_transaction_dist.png')

    # Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(eda_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/heatmap.png')

    return eda_df


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    # Copy DataFrmae
    encoder_df = dataframe.copy(deep=True)

    for category in category_lst:
        column_lst = []
        column_groups = dataframe.groupby(category).mean()['Churn']

        for val in dataframe[category]:
            column_lst.append(column_groups.loc[val])

        if response:
            encoder_df[category + '_' + response] = column_lst
        else:
            encoder_df[category] = column_lst

    return encoder_df


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
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
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


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
    pass


if __name__ == "__main__":
    # Import data
    BANK_DF = import_data(file_path='./data/bank_data.csv')
    # perform EDA
    EDA_DF = perform_eda(dataframe=BANK_DF)
