# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.


## Summary
The dataset contains about bank-marketing data, where the marketing dept made the call to the customer. The last column is labeld as 0 or 1.

This is a classification problem. We use both AutoML and Hyperdrive to solve this problem. We used TabularDatasetFactory to get the data, cleaned it and then ran bunch of model to get the best model.

The performance of both models are good. Though Automl model got 91.60% Accuracy and our hyperdrive got 91.59% Accuracy.

## Architecture Diagram
<p align="center"><img src="1.png"></p>
This is the diagram for both automl and hyper drive.

## Scikit-learn Pipeline
We used TabularDatasetFactory to load the data and clean data using pandas. Both had been done in train.py Then we use SKLearn to run the train.py and feed hypderdrive config. HypderDrive later used different type of C and max_iter to find the best model.


By using parameter sampling i tuned hyperparameter and tried to find the suitable spot of the model. I just didn't do an memory hungry search such as Grid Search. I used both uniform and discreate search for two parameters sampler. Which will be robust and also give an acceptable result.


We used early stopping policy to ensure that our model terminate poorly performing runs and save resources. We used BanditPolicy which ensure that any run will terminated if the primary matrics(Accuracy) of a run is less than the slack factor of best run. We use 0.1 for slack factor

## AutoML
The AutoML is configured to allow 5 croos validation for the classification task.  The model output is a pipeline with 2 steps. 
- datatransformer
- prefittedsoftvotingclassifier
<br>
for the classifier hyperparameters, penalty is set to l2 and max_iter is set to 1000. 

## Pipeline comparison
Both models are very good. Though AutoML got a slightly better because of trying out lots of models. And found out that VotingEnsemble is the best model(91.60% Accuracy). Where in Hyperdrive it only used logistic regression model.

## Future work
I would like to add more hyper parameters to tweak the model more in depth to see the result. 
Also I will move the cleaning data process into different pipeline and will try out different normalization technique.
