## Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

### Project Description

In this project, you will implement your learnings to identify credit card customers that are most likely to churn. The completed project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

This project will give you practice using your skills for testing, logging, and best coding practices from this lesson. It will also introduce you to a problem data scientists across companies face all the time. How do we identify (and later intervene with) customers who are likely to churn?

#### DataFrame

![kaggle](https://user-images.githubusercontent.com/9282421/199313132-f32be46c-15a8-4231-9b95-aa5b01198aad.png)

## Files and data description

### Overview of the files and data present in the root directory

![file-overview](https://user-images.githubusercontent.com/9282421/199308967-337d6098-ecc2-4ebb-b7a8-bb0c5a8f9184.png)

### Running Files

We will use git and [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):

1. Clone the repository

2. Create a conda environment

3. Activate the conda environment

### Install the linter and auto-formatter

---

```
~/project$ pip install pylint
~/project$ pip install autopep8
```

### Run python files

---

```
~/project$ python churn_library.py
~/project$ python python_script_logging_and_tests.py
```

### check the pylint score using the below

---

```
~/project$ pylint churn_library.py --disable=invalid-name,too-many-arguments

~/project$ pylint churn_script_logging_and_tests.py --disable=invalid-name,too-many-arguments
```

### To assist with meeting pep 8 guidelines, use autopep8 via the command line commands below

---

```
~/project$ autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py

~/project$ autopep8 --in-place --aggressive --aggressive churn_library.py
```

## References

[Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code)
