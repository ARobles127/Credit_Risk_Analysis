# Credit_Risk_Analysis


## Overview of the analysis:

Machine learning is a powerful method  that helps data scientists analyze and automate models that can learn from data, identify patterns and make decisions. For this project we are going to analyze a dataset and build and test models that help a financial institution measure credit risk and improve their capabilities to select lenders based on  analysis performed by these machine learning models.
To help accomplish our task we will use Python  and Jupyter to test on different machine learning models. 



## Results: 

The following images contained the results found after running six different machine learning models.


### Naive Random Oversampling

Random oversampling involves randomly duplicating examples from the minority class and adding them to the training dataset. Based on our findings this model may not be the best one for preventing high risk loan applications because the model's accuracy, 0.641, is low. The precision and recall are not good enough  to state that the model will be good at classifying low from high risk loan applications.

![NRO_AS.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/NRO_AS.png) 

![NRO_CM.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/NRO_CM.png) 

![NRO.CRI.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/NRO_CRI.png) 


### SMOTE Oversampling

Results from this model are similar to the previous module. The accuracy score is low, 0.63. The confusion matrix results are the same as the Naive Random Oversampling. For this reason this model would not be efficient at classifying high risk applications. 


![SO_AS.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/SO_AS.png) 

![SO.CM.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/SO_CM.png) 

![OS_CRI.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/OS_CRI.png) 


### Undersampling

Random undersampling involves randomly selecting examples from the majority class to delete from the training dataset. Results from this model suggest that the precision for the low risk applications is low, 0.529, indicating a large number of false positives, which indicates an unreliable positive classification. The recall is also low for the low risk applications, which is indicative of a large number of false negatives. The F1 score is also low (0.01) .In summary, this random forest model is not good at classifying fraudulent loan applications because the model's accuracy, 0.529, and F1 score are low.

![U_AS.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/U_AS.png) 

![U_CM.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/O_CM.png) 

![U_CRI.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/U_CRI.png) 


### Combination (Over and Under) Sampling

Interesting results may be achieved by combining both random oversampling and undersampling. In this case our results are slightly variant and suggest that the accuracy score is slightly better at predicting low risk applications, 0.624. This model fails at classifying high risk applications with an f1 score of 0.02. Still the model accuracy numbers are not significant enough consider this model as an efficient one. 

![CS_AS.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/CS_AS.png) 

![CS_CM.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/CS_CM.png) 

![SC_CRI.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/CS_CRI.png) 


### Balanced Random Forest Classifier

In general, the metrics (precision, recall, and F1 score)of accuracy score are slightly improved over those of random oversampling. 

![BRF_AS.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/BRF_AS.png) 

![BRF_CM.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/BRF_CM.png) 

![BRF_CRI.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/BRF_CRI.png) 



### Easy Ensemble AdaBoost Classifier
AdaBoost classifier offers a better prediction than all five previous models. The accuracy score is 0.925, while the prediction rates are more accurate which means that this model could be better at predicting high risk applications. 

![EEA_AS.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/EEA_AS.png) 

![EEA_CM.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/EEA_CM.png) 

![EEA_CRI.png](https://github.com/ARobles127/Credit_Risk_Analysis/blob/main/Images/EEA_CRI.png) 



## Summary: 

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, it is crucial for a financial institution to rely on a model that helps predict high risk loan applications and at the same time the model has to be good at classifying low and high risk applicants with low error mirgings. For this reason we would  recommend that a model like Easy Ensemble AdaBoost Classifier is the best approach to predict credit risk. 
