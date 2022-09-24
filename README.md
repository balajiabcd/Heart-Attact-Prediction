# Heart-Attact-Prediction








## Libraries Used:  
  
1. Numpy  
2. Pandas  
3. Matplotlib  
4. Seaborn  
5. SciKit Learn   









## Data Set:  
  
You can Download the data set used in this project here at this link.  
"https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset"  








## Flatform:  
  
For this project I used Kaggle. Which you can found here: kaggle.com  









## About data:  

data consists total of 303 patients's diagnossis results. Out of 165 patients found to have high chance of heart failure.  



After encoding the variables "cp" and "restecg", we can remove one of each encoded variables to remove duplicacy. But only one encoded variable was removed afer observing the results. The heat map of ramaining variables is shown here.  
![Heatmap-after-processing](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/images/before.png)  


With the histplots between the variables we can see the there is distinction between data of patients who has heart failure and data of patients who don't have it.  
![Histplot](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/images/histplot.png)  


With the pai between the variables, we can see that it will be a bit heard to separate the patients based on only any 2 variables.
![Scattereplot](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/images/pairplot.png)












## Note:  
Here to get the accuracy of classification model we used MAE(which is one of evaluation matrics of Regression model) as an indirect way of finding acuuracy of model.  












## Model Result:  


After trining and Deploying the different classification models and one regression model:  
Naive Bayes model, logistic regression models gave absolute ~89.5% accurate prediction on testset.  

 










