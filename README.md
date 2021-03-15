# Neural_Network_Charity_Analysis

## Overview

Alphabet Soup is a non-profit foundation dedicated to helping organizations that protect the environment, improve people's health, and unifies the world. From their business team, we received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are several columns that capture metadata about each organization. The purpose of this project is to analyze the impact of each donation and assess potential recipients. We want to make sure that the foundation's money is being used effectively. Therefore, the goal is to identify which organizations are worth the donations and which ones are high risks. 

## Analysis Results

With machine learning and neural networks, we use the features in the provided dataset to help create binary classifiers that can predict whether applicants will be successful if funded by Alphabet Soup. 

#### Data Preprocessing

In the initial model, two variables were removed from the input data: EIN and NAME. These two variables are identification columns. Having them in the dataset creates too much noise, especially with the amount of organization names being many and it will not allow the machine learning to be flexible in case we are to repurpose the model if by chance we are given the same but updated dataset containing new organization names on the list.

In the optimized model, additional variables were dropped from the DataFrame due to its lack of contributions as a feature to the main analysis goal. We want to be able to narrow the prediction of successful applicants and identify the ones that are high risks. These dropped variables are non-beneficial as we should only need to know the application type of each recipient, what sector of the industry are they affiliated to, the government classification and/or organization they belong to, and their use case for the funding. Below is a breakdown of the variables.

##### Target Variable:
* IS_SUCCESSFUL - this is the variable that identifies whether the money was used effectively.

##### Feature Variables that has been placed in category bins:
* APPLICATION_TYPE - this is Alphabet Soup's classification of application type.
* AFFILIATION - identifies the sector of the industry whether they are independent, company sponsored, or other.
* CLASSIFICATION - relates to government organization classification.
* USE_CASE - use for the funding whether its for preservation, product development, community service, or other
* ORGANIZATION - types of organization they belong to whether trust, association, or other.

##### Dropped Variables:
* EIN and NAME - are the identification columns.
* STATUS - just shows either active or not.
* INCOME_AMT - is an income classification.
* SPECIAL_CONSIDERATIONS - indicates if special consideration is needed for application.
* ASK_AMT - is the funding amount requested.

Using TensorFlow, we have been tasked to optimize the original model in order to achieve a target predictive accuracy higher than 75%. However, I was unable to achieve an accuracy higher than 75%. I made many efforts and provided below are the top three model attempts I had performed.

#### Compiling, Training, and Evaluating the Model

Initial Model
* Hidden Layer 1 with 80 neurons and activation function at "relu"
* Hidden Layer 2 with 30 neurons and activation function at "relu"
* Output Layer with activation function at "sigmoid"
* Set at 100 epochs with the model's weights are saved every 5 epochs.
* Resulting in an accuracy score of 69% (0.6866)

![OriginalModel_AccuracyScore](https://github.com/Lora-Borja/Neural_Network_Charity_Analysis/blob/main/images/OriginalModel_AccuracyScore.PNG)

##### 1st Attempt

![OptimizedModel_Attempt1_Result](https://github.com/Lora-Borja/Neural_Network_Charity_Analysis/blob/main/images/OptimizedModel_Attempt1_Result.PNG)

* Resulting in an accuracy score of 69% (0.6924) - just a tiny bit lower than the initial model!

![OptimizedModel_Attempt1_AccuracyScore](https://github.com/Lora-Borja/Neural_Network_Charity_Analysis/blob/main/images/OptimizedModel_Attempt1_AccuracyScore.PNG)

##### 2nd Attempt

![OptimizedModel_Attempt2_Result](https://github.com/Lora-Borja/Neural_Network_Charity_Analysis/blob/main/images/OptimizedModel_Attempt2_Result.PNG)

* Resulting in an accuracy score of 72% (0.7174)

![OptimizedModel_Attempt2_AccuracyScore](https://github.com/Lora-Borja/Neural_Network_Charity_Analysis/blob/main/images/OptimizedModel_Attempt2_AccuracyScore.PNG)

##### 3rd Attempt

![OptimizedModel_Attempt3_Result](https://github.com/Lora-Borja/Neural_Network_Charity_Analysis/blob/main/images/OptimizedModel_Attempt3_Result.PNG)

* Resulting in an accuracy score of 72% (0.7186) - just a bit higher than the 2nd attempt!

![OptimizedModel_Attempt3_AccuracyScore](https://github.com/Lora-Borja/Neural_Network_Charity_Analysis/blob/main/images/OptimizedModel_Attempt3_AccuracyScore.PNG)

## Summary

Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem and explain your recommendation.
