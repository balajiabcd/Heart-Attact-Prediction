# Heart-Attact-Prediction
  
Find and install the libraies used in this project in requirments.txt file.   
And download the data set used in this project here at this link.  
"https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset"  



## Flatform:  
  
For this project was done completely done in VS code, and Diplyoyed the app in to Amazon Web Services (AWS)   


## About data:  

The data consists total of 303 patients's diagnossis results. Out of 165 patients found to have high chance of heart attack and remaining patients have less chance of heart attack. In the obtained data from kaggle, there were no missing values. Data consists of multiple variables, some are quantitative and some are qualitative. These qulitative varialble are encoded first encoded in to multiple variables. After encoding the variables "cp" and "restecg", we can remove one of each encoded variables to remove duplicacy. 

Then heat map was plotted for the variables, after observing the map, one encoded variable was removed. The heat map of ramaining variables is shown here.  
![Heatmap-after-processing](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/static/images/heatmap.png)  


With the histplots of the variables we can see the there is distinction between data of patients who has heart failure and data of patients who don't have it.  
![Histplot](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/static/images/histplot.png)  


With the pairplot between the variables, we can see there exists some clusters in the various plots. Hence, we can expect high accuracy.
![Scattereplot](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/static/images/pairplot.png)

## Note:  
Here to get the accuracy of classification model we used MAE(which is one of evaluation matrics of Regression model) as an indirect way of finding acuuracy of model. After trining and Deploying the different classification models and one regression model: Naive Bayes model, logistic regression models gave absolute ~89.5% accurate prediction on testset.    

 










