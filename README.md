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

data consists total of 303 patients's diagnossis results. Out of 165 patients found to have heart failure.  


Upon observing the heat map between the independent variables in data, only 2 variables "restecg_0" and "restecg_1" have strong correlation.    
![Heatmap-before-processing](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/Imges-repo/Heatmap-before-processing.png)  


Hence, we can remove one of these variables from our model.  
![Heatmap-after-processing](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/Imges-repo/Heatmap-after-processing.png)  


With the histplots between the variables we can see the there is distinction between data of patients who has heart failure and data of patients who don't have it.  
![Histplot](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/Imges-repo/histplot.png)  


With the scaterplot between the variables, we can see that it will be a bit heard to separate the patients based on only any 2 variables.
![Scattereplot](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/Imges-repo/subplots.png)












## Note:  
Here to get the accuracy of classification model we used MAE(which is one of evaluation matrics of Regression model) as an indirect way of finding acuuracy of model.  












## Model Result:  


After trining and Deploying the different classification models and one regression model:  
Randomforest model, Decission tree model, Naive Bayes model, linear regression models gave absolute 100% accurate prediction of testset.  

![Randomforest model](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/Imges-repo/randomforestmodel-results.png)  

![Decission tree model](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/Imges-repo/Decissiontree-model.png)  

![Naive Bayes model](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/Imges-repo/NBmodel-results.png)  

![linear regression ](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/Imges-repo/regressionmodel-results.png)  










