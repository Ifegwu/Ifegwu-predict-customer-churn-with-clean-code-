'''
Module contains test for churn customer analysis

Author: Ifegwu Daniel Agbanyim

Date: 28th October 2022
'''

# Import libaries
import os
import logging
from pickle import TRUE
import churn_library as clib

# logging configuration
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = clib.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        logging.info('Rows: %d\tColumns: %d',
                     dataframe.shape[0], dataframe.shape[1])
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''

    dataframe = clib.import_data("./data/bank_data.csv")

    try:
        clib.perform_eda(dataframe=dataframe)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as err:
        logging.error('Column "%s" not found', err.args[0])
        raise err

    # Assert that `churn_dist.png` is created
    try:
        assert os.path.isfile("./images/eda/churn_dist.png") is True
        logging.info('File %s was found', 'churn_dist.png')
    except AssertionError as err:
        logging.error('No such file on the disk')
        raise err

    # Assert that `customer_age_dist.png` is created
    try:
        assert os.path.isfile("./images/eda/customer_age_dist.png") is True
        logging.info('File %s was found', 'customer_age_dist.png')
    except AssertionError as err:
        logging.error('No such file on the disk')
        return err

    # Assert that `dataframe.png` is created
    try:
        assert os.path.isfile("./images/eda/dataframe.png") is True
        logging.info('File %s was found', 'dataframe.png')
    except AssertionError as err:
        logging.error('No such file on the disk')
        return err

    # Assert that `heatmap.png` is created
    try:
        assert os.path.isfile("./images/eda/heatmap.png") is True
        logging.info('File %s was found', 'heatmap.png')
    except AssertionError as err:
        logging.error("No such file on the disk")
        return err

    # Assert that `marital_status_dist.png` is created
    try:
        assert os.path.isfile("./images/eda/marital_status_dist.png") is True
        logging.info('File %s was found', 'marital_status_dist.png')
    except AssertionError as err:
        logging.error('No such file on the disk')
        return err

    # Assert that `total_transaction_dist.png` is created
    try:
        assert os.path.isfile(
            "./images/eda/total_transaction_dist.png") is True
        logging.info("File %s was found", "total_transaction_dist.png")
    except AssertionError as err:
        logging.error("No such file found on the disk")
        return err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    test_import()
    test_eda()
