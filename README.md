
# Assignment 2

## Instructions 

One of the really hot trends in machine learning at the moment is automated machine learning, or automl. There are several tools, open source and commercial, e.g. H2O that are focusing on this. The key idea behind this movement, and the tools in question, is that once we have data prepared for modeling (and remember, this can often constitute a resource-intensive task that is difficult to avoid) we often follow a scripted procedure of preparing training and test sets, trying out different models, tuning these models with methods such as cross-validation and then comparing these models to pick the final results. Up till now, this has usually involved training models sequentially by hand, and then manually issuing commands to begin training the next model etc… We are now moving to a phase where a lot of the repetitive work that we have been doing as data scientists is beginning to be automated.

With that in mind, the objective of this assignment is to introduce you to this exciting field and mode of work in what we believe to be the best way possible – by actually building your own automl system in Python. This may sound scary, especially for those of you who have not had extensive experience in programming before, but this assignment will walk you through the entire process.

You are required to complete this workbook by filling in all the places in the code that are marked "YOUR CODE HERE". You must fill all of these correctly in order to get full marks. Do NOT change any other aspect of this workbook otherwise you may lose points. Please also make sure to complete the following section by giving us your name and student ID. Once you are ready, submit your completed notebook to Moodle.

Good luck and enjoy!

## Student Identification

Student FULL NAME: Eva Giannatou

Student ID: BAFT1616

## Setup

Run the following cell in order to load the packages you will need for this assignment.


```python
# Let's start off with all the basic imports
# Make sure you run this cell!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics
import random
import warnings
warnings.filterwarnings('ignore')
```

## Main Objective

In this assignment, we are going to build a function that will take in a Pandas data frame containing data for a **binary classification** problem. Our function will **try out and tune** many different models on the input data frame it receives and at the end it is going to return the model it thinks is best, as well as an expectation of its performance on new and unseen data in the future. To achieve this mighty task we are going to build several helper functions that our main function is going to have access to.

## Question 1 (2 points)

We've seen than building models in scikit-learn requires us to specify a matrix or data frame X with just the input features, and an array or Series object y with just the values of the target variable. To that end, let's first build a simple function, ```extract_x_and_y()```, whose role will be to take in a Pandas data frame and the name of the target column (i.e. the column with our desired y values). This should then simply return:

- ```X```: the input data frame without the target column
- ```y```: the target column on its own. 

X and y should be returned as a tuple.


```python
def extract_x_and_y(df, y_column):
    return(df.drop(y_column, axis=1), df[y_column])
```

## Question 2 (1 point)

Ok, we now have X and y. Before we start training our models, however, we need to make sure that we split our input data into a training set and a test set. As we have X and y, we will need X_train and y_train along with X_test and y_test. Write a *very simple* function that takes as input:

- ```X``` : The input data frame without the target column
- ```y```: The target variable on its own
- ```test_size```: The size of the test set, with a default of 20%
- ```random_state```: An integer for reproducible sampling, with a default of 42.

and returns as a tuple:

- ```X_train```
- ```y_train```
- ```X_test```
- ```y_test```


```python
def split_x_and_y(X, y, test_size = 0.2, random_state = 42):
        
    np.random.seed(random_state)
    
    msk = np.random.rand(len(y)) < 1-test_size
    X_train=X[msk]
    X_test=X[~msk]
    y_train=y[msk]
    y_test=y[~msk]
    return (X_train, y_train, X_test, y_test)
```

## Question 3 (2 points)

In this question we are going to take a step back from function writing and think about the models we are going to build in our automated machine learning script. We need to have a way to tell our script to iterate through a predefined list of models. We've seen that in scikit-learn, the basic procedure for training any model is a three step affair:

- Loading the required class (we have imported relevant submodules in an earlier cell so you don't have to worry about this step)
- Creating an intial variable by calling the model class' *constructor*. This is a function for initializing the model, which may or may not (depending on the model type) take certain parameters as input
- Executing the ```fit()``` method on that class.

Notice we call the same method i.e. ```fit()``` on any model we use. Now, our objective is to build a list of models that our automl function can iterate over one by one. To do this, we can initialize all the different models we want by calling their constructor class, then save them as variables and finally collect them all in a list. Each model, however, comes with certain parameters we need to tune. We saw that the ```GridSearchCV()``` function can take an initialized model and a dictionary with parameters for that model and implement cross validation to tune that model. So, for every model we need to keep track of what parameters we need to tune for that model. Moreover, we should also keep track of a human readable name for that model which is useful when printing out information for the user. Putting this together, what we really want to do is build a dictionary data structure **for each** model that we are going to train. Then we are simply going to store all these dictionaries in a list and our main function will iterate through this list.

Here are the three keys we want in the dictionary data structure that describes each of our models:

- ```name``` : A string with a human readable name for the model
- ```class``` : A call to the class for initializing our model. This is what we usually write as a second step in our three step process for training a model with scikit learn. Instead of saving the result to a variable, we simply save the result in this key. When initializing your model, do not specify any parameters you want to tune with cross validation.
- ```parameters``` : The value of this key is a dictionary where the keys are parameters we want to tune with cross validation. The format of this is exactly what we would pass in to the ```parameters``` input parameter for  ```GridSearchCV()```. The Jupyter notebook on decision trees has simple examples of this.

Ok, enough reading. Let's test your understanding by constructing a dictionary that will describe your first model. We will do it for the Logistic Regression model with LASSO regularization. Here is what you need to do:

- Create a dictionary variable and call it ```loglas```
- Create the ```name``` key, and give it the string value "Logistic Regression with LASSO"
- Create the ```class``` key by calling the constructor class for Logistic Regression with Lasso which is ```sklearn.linear_model.LogisticRegression()```. Notice we are using the full path in sklearn to this class because we have not written import statements for each model as we have been doing so far. You should pass a ```penalty``` parameter with the value ```'l1'``` to this constructor to let scikit learn know we are using the LASSO operator.
- Create the ```parameters``` key. For the Logistic Regression model, the only parameter we need is ```C``` (this is related to lambda that we met when we studied logistic regression). Thus you should create a dictionary with ```C``` as the only key. The value of the ```C``` key needs to be a list of values that cross validation will try out. Please use this list of values: 0.001, 0.01, 0.1, 1, 10 and 100. 


```python
loglas =  {"name":"Logistic Regression with LASSO", 
           "class": sklearn.linear_model.LogisticRegression(penalty='l1'),
           "parameters":{"C": [0.001, 0.01, 0.1, 1, 10, 100]}} 
loglas
```




    {'class': LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
               intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
               penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
               verbose=0, warm_start=False),
     'name': 'Logistic Regression with LASSO',
     'parameters': {'C': [0.001, 0.01, 0.1, 1, 10, 100]}}



## Question 4 (4 points)

In the previous question we described the specific dictionary data structure that will hold the definitions of the algorithms we are going to run. In this question we are going to build the complete list containing all the dictionaries for all the different models. This is an excellent place for you to experiment with new classification algorithms if you want to look them up in the scikit-learn documentation! If you want to add some more models please feel free to do so but at the bare minimum however, we want your list of models to contain definitions for the following:

| Model || Parameters   |
|------||------|
| K Nearest Neighbors Classifier || n_neighbors : The numbers 1 through 12 |
| Support Vector Classifier with Linear Kernel || C : The values 0.001, 0.01, 0.1, 1, 10, 100 |
| Support Vector Classifier with Radial Kernel || C : The values 0.001, 0.01, 0.1, 1, 10, 100 <br>gamma : The values 0.001, 0.01, 0.1, 1, 10, 100  |
| Logistic Regression with LASSO || C : The values 0.001, 0.01, 0.1, 1, 10, 100 | 
| Stochastic Gradient Descent Classifier || max_iter : The values 100, 1000 <br> alpha : The values 0.0001, 0.001, 0.01, 0.1 |
| Decision Tree Classifier || max_depth : The numbers 3 through 15 |
| Random Forest Classifier || n_estimators : The values 10, 20, 50, 100, 200 |
| Extremely Randomized Trees Classifier || n_estimators : The values 10, 20, 50, 100, 200 |

Your task: Create a function ```specify_models()``` that takes no parameters at all and returns **a list of model definitions** for each of the above classifiers, where each model definition is the dictionary structure described in Question 3.

*Hint*: To avoid having to do import statements for each classifier class, follow the example from Question 3 where we specify the full path of the class in the scikit-learn package e.g. ```sklearn.neighbors.KNeighborsClassifier()``` is what you want for K Nearest Neighbors. You still of course need to load certain modules from sklearn at the start for this work, but you will note that we have already done this for you at the start of this notebook.

*Hint*: To find out what the right class is for each classifier in the table above, look at your notes and/or do a quick search on Google or the scikit-learn documentation. It should take you just a few seconds to find the right class name.


```python
def specify_models():
    loglas =  {"name":"Logistic Regression with LASSO", 
           "class": sklearn.linear_model.LogisticRegression(penalty='l1'),
           "parameters":{"C": [0.001, 0.01, 0.1, 1, 10, 100]}} 
    
    knn =  {"name":"K Nearest Neighbors Classifier", 
           "class": sklearn.neighbors.KNeighborsClassifier(),
           "parameters":{"n_neighbors": list(range(1,13))}} 
    
    svcLinear =  {"name":"Support Vector Classifier with Linear Kernel", 
           "class": sklearn.svm.LinearSVC(),
           "parameters":{"C": [0.001, 0.01, 0.1, 1, 10, 100]}}     
    
    svcRadial =  {"name":"Support Vector Classifier with Radial Kernel", 
           "class": sklearn.svm.SVC(kernel='rbf'),
           "parameters":{"C": [0.001, 0.01, 0.1, 1, 10, 100], "gamma" : [0.001, 0.01, 0.1, 1, 10, 100]}}       
    
    sgd =  {"name":"Stochastic Gradient Descent Classifier", 
           "class": sklearn.linear_model.SGDClassifier(),
           "parameters":{"max_iter": [100, 1000], "alpha" : [0.0001, 0.001, 0.01, 0.1]}}     
    
    tree =  {"name":"Decision Tree Classifier", 
           "class": sklearn.tree.DecisionTreeClassifier(),
           "parameters":{"max_depth": list(range(3,16))}}     
    
    rfc =  {"name":"Random Forest Classifier", 
           "class": sklearn.ensemble.RandomForestClassifier(),
           "parameters":{"n_estimators": [10, 20, 50, 100, 200]}}   
    
    rfcextra =  {"name":"Extremely Randomized Trees Classifier", 
           "class": sklearn.ensemble.ExtraTreesClassifier(),
           "parameters":{"n_estimators": [10, 20, 50, 100, 200]}} 
    
    model_dict = [loglas, knn, svcLinear, svcRadial, sgd, tree, rfc, rfcextra]
    return(model_dict)
```

## Question 5 (4 points)

What we have right now a list of dictionaries. Each dictionary essentially has the ingredients for us to train a model and tune the right parameters for that model. So, what we need now is a function, ```train_model()``` that takes in the following parameters:

- ```model_dict``` : We will pass in the dictionaries from the list you just created one by one to this parameter
- ```X```: The input data
- ```y```: The target variable
- ```metric``` : The name of a metric to use for evluating performance during cross validation. Please give this parameter a default value of 'f1' which is the F measure.
- ```k``` : The number of folds to use with cross validation, the default should be 5

This function should essentially just call ```GridSearchCV()``` by correctly passing in the right information from all the different input parameters. The function should then return:

- name : The human readable name for the model type that was trained
- best_model : The best model that was trained 
- best_score : The best score (for the metric provided) that was found

As usual, please return these three variables as a simple tuple.

*Hint*: Look at the online sklearn documentation for ```GridSearchCV()``` to find out how to pass in the name of a metric to use for scoring and how to get the best score that was seen for this metric across the different parameters. Do not fret that you have to do this. Gaining familiarity with the online documentation is a core part of working with open source tools like sklearn.


```python
from sklearn.model_selection import GridSearchCV

def train_model(model_dict, X, y, metric = 'f1', k = 5):
    
    
    
    keys = []
    for key in model_dict:
        keys.append(model_dict[key])

    clf = GridSearchCV(estimator=keys[1], param_grid=keys[2], cv=k, scoring=metric)
    clf.fit(X, y)

    best_score = clf.best_score_
    best_model = clf.best_estimator_
    name = keys[0]

    return(name, best_model, best_score)
```

## Question 6 (4 points)

We're now ready to write the central component of our automl system! We need to iterate through all of our models, train them on the training data and report back all the results. Concretely, you should define a function ```train_all_models()``` that takes in:

- ```models``` : The list of the dictionaries that describes the models we will train
- ```X```: The input data that will be used for training
- ```y```: The values of the target variable for the training set
- ```metric``` : The name of a metric to use for evaluating performance during cross validation. Please give this parameter a default value of 'f1' which is the F measure 
- ```k``` : The number of folds to use with cross validation, the default should be 5

This function should apply the ```train_model()``` function you created in the previous question to every model in the ```models``` list. The resulting tuples from each call to ```train_model()``` should be stored in a list **and this list should be returned in sorted order by descending score**. This means that in the output list, the first tuple returned will have the best model found.

*Note: We will assume just for this exercise that the metric used for evaluating the models will give high values for good models and low values for bad models. A better implementation would sort the list based on what type of metric we used. I leave this as an exercise for the avid student to pursue in their own time.*


```python
def train_all_models(models, X, y, metric = 'f1', k = 5):

    trained_models = []
    for index in range(len(models)):

        name, best_model, best_score = train_model(models[index], X, y, metric = 'f1', k = 5)

        trained_models.append([name, best_model, best_score])
        trained_models.sort(key=lambda k:(k[2]), reverse=True)


    return(trained_models)
```

## Question 7 (8 points)

We now have everything we need to write our automated binary classifier, so let's create it and call it ```auto_train_binary_classifier()```. This function should take:

- ```df```: An initial Pandas data frame
- ```y_column``` : The column in the data frame that has the target binary variable
- ```models``` : The list of the dictionaries that describes the models we will train
- ```test_size```: The size of the test set, with a default of 20%
- ```random_state```: An integer for reproducible sampling, with a default of 42.
- ```metric``` : The name of a metric to use for evaluating performance during cross validation. Please give this parameter a default value of 'f1' which is the F measure 
- ```k``` : The number of folds to use with cross validation, the default should be 5

This function should then take the following steps:

1. Split your initial dataframe into your data X and your binary target variable y (you wrote a function for this above)
2. Split the data into training and testing (you wrote a function for this above)
3. Train all the models in the models list on the **training data** (you wrote a function for this above)
4. Retrieve the best model from the models trained, its name, and its training set performance (score)
5. Run the best model on the **test set** and retrieve the test set performance (score)

Once finished, your function should return the following:

- The name of the best model you found (human readable string)
- The best model you found
- The training set performance (measured using the metric you provided)
- The test set performance obtained by using classification accuracy as the metric

Simply return these four as a tuple.

*Note: There are so many avenues for improving this function that could make interesting exercises for the avid student to work on but are outside the scope of this assignment. These include:*

- *Reporting test set performance using the same metric as the one used for cross validation*
- *Ensuring that we also experiment with scaling/normalizing our input data because certain models (SGD, SVMs etc...) tend to work better when this is the case*
- *Ensuring that once we've decided on the best model, we  retrain on the entire data set provided using the parameters of the best model we found.*
- *Incorporating feature selection techniques that might be appropriate to different model types*


```python
def auto_train_binary_classifier(df, y_column, models, test_size = 0.2, random_state = 42, 
                                 metric = 'f1', k = 5):
    
    X,y = extract_x_and_y(df, y_column)
    X_train, y_train, X_test,  y_test = split_x_and_y(pd.DataFrame(X), pd.DataFrame(y) , test_size = 0.2, random_state = 42)
    models = specify_models()
    trained_models = train_all_models(models, pd.DataFrame(X_train), pd.DataFrame(y_train), metric = 'f1', k = 5)
    bm = trained_models[0]
    best_model = list(filter(lambda d: d['name'] in [bm[0]], models))

    test_set = train_all_models(best_model, X_test, y_test, metric = 'f1', k = 5)

    test_set_performance = test_set[0][2]

    return(bm[0], bm[1], bm[2], test_set_performance)
```

You should realize that I've kept the list of models as an input to your ```auto_train_binary_classifier()``` function. This allows you to change the implementation of your automated tool to add and remove certain models from consideration by simply changing your implementation of ```specify_models()``` without changing the implementation of your main ```auto_train_binary_classifier()``` function.

Congratulations! You've built yourself a very simple but useful automated tool for performing binary classification. As indicated by some of the notes in the preceding questions, you'll have found that there are plenty of avenues for improvement. Nonetheless, I hope that you found this exercise motivating and a valuable glimpse into how you might begin to apply practical techniques to become more efficient at finding a good solution to a modeling problem. Remember - a lot of the work we do in data science and analysis lies in the intial data preparation and feature engineering phase which this tool obviously does not cover at all. Current research is aimed at pushing the boundaries of automation in that direction too and I encourage you to find out more about this if you are interested.

## Testing

This section is an opportunity for you to test what you have implemented in this assignment. There are no more questions in this assignment, this section is only there to help you. In the code below, we've loaded up a data set into a Pandas dataframe and we call your ```auto_train_binary_classifier()``` function to see the result. Use this as an opportunity to see if your function returns an output that you expect.


```python
from sklearn.datasets import load_breast_cancer, load_iris
cancer = load_breast_cancer()
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df['target'] = pd.Series(cancer.target)

# The next commands will only work once you've implemented these functions above.
models = specify_models()
best_model_name, best_model, train_set_score, test_set_score = auto_train_binary_classifier(cancer_df, 'target', models)
print(best_model_name)
print(best_model)
print(train_set_score)
print(test_set_score)
```

    Logistic Regression with LASSO
    LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)
    0.967387701671
    0.976844739327

