#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split, cross_val_score
import os
from itertools import cycle
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, linear_model, metrics, datasets, multiclass, svm
import seaborn as sns
import numpy.random as nr
import joblib

# In[8]:


# import zipfile
# with zipfile.ZipFile('consumer_complaints.csv.zip', 'r') as zip_ref:
#     zip_ref.extractall()


# In[9]:


Data = pd.read_csv('consumer_complaints.csv', encoding='latin-1')


# In[10]:


import pandas as pd 
pd.get_option('display.max_columns', 500)
pd.get_option('display.max_rows', 100)
pd.get_option('display.max_colwidth', -1)


# In[11]:


Data


# In the above data we have to just classify products based on narrative given by customers so, we need 2 column i.e. `product` and `consumer_complaint_narrative`

# ## Data understanding 

# In[12]:


Data.dtypes


# In[13]:


pd.notnull(Data['consumer_complaint_narrative']).value_counts()


# Take data which contain atleast 1 word which is to useful while model building, From the above output it shows `66806` are not null rows so we have to take only those rows. 

# In[14]:


Data = Data[['product','consumer_complaint_narrative']]
Data = Data[pd.notnull(Data['consumer_complaint_narrative'])]
Data


# In[15]:


Data.shape


# In[16]:


# check the distribution of complaint by category
Data.groupby('product').consumer_complaint_narrative.count()


# **Note:** Imbalance Dataset, `Other financial service`, `Money transfers`, `Payday loan`, `Prepaid card` are having less than `1000` rows denoting imbalance dataset

# Let's Analyse it Graphically

# ## Splitting the data

# In[17]:


fig = plt.figure(figsize=(8,6))
Data.groupby('product').consumer_complaint_narrative.count().plot.bar()
plt.show()


# ## Converting Text to Features
# 
# The procedure of converting raw text data into machine understandable format(numbers) is called feature engineering of text data. Machine learning and deep learning algo performance and accuracy is fundamentally dependent on the type of feature engineering techniques used.

# ## TF_IDF Vectorizer
# 
# TF_IDF is the most applied feature engineering technique for processing textual kind data by many machine learning expert and data scientist.
# 
# Term Frequency(TF): -> Term frequency is simply the ratio of the count of a word present in a sentence, to the length of the sentence.
#                     -> TF is basically capturing the importance of the word irrespective of the length of the document.
#                     -> ex: a word with the frequency of 3 with the length of sentence being 10 is not the same as when the word length of sentence being 100 words. It should get more importance in the first scenario; that is what TF does.
#   
# Inverse Document Frequency(IDF): -> IDF of each word is the log of the ratio of the total number of rows to the number of rows in a 
#                                     particular document in which that word is present.
#                                  -> IDF will measure the rareness of a term. word like 'a' and 'the' show up in all the documents of                                           corpus, but the rare words is not in all the documents.
#                                 
# TF-IDF is the simplest product of TF and IDF so that both of the drawbacks are addressed, which makes predictions and information retrieval relevant.

# In[18]:


tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern='\w{1,}', max_features=5000)    # default {max_df=1.0(float) so its proportion of word contain in all documents,
# simply if a words contain in all the document it is 1.0}, {min_df=1(int) so iteger directly denotes count of words in a document shows atleast 1 word contains in a documents then only it is consider}
tfidf_vect.fit(Data['consumer_complaint_narrative'])
Features = tfidf_vect.transform(Data['consumer_complaint_narrative'])

encoder = preprocessing.LabelEncoder()
Labels1 = encoder.fit_transform(Data['product'])

# Binarize the output
#Labels = np.array(preprocessing.label_binarize(Labels, classes=[0,1,2,3,4,5,6,7,8,9,10]))


# In[19]:


print(Features[0], Labels1)


# Next, execute the code in the cell below to split the dataset into test and training set. Notice that usually, 25% of the 100% cases are being used as the test dataset. 

# In[20]:


train_x, valid_x,  train_y, valid_y = train_test_split(Data['consumer_complaint_narrative'],Data['product'])    # Default it will split 25 by 75% means 25% test case and 75% training cases


# In[21]:


train_x


# In[22]:


encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
print(train_y)

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern='\w{1,}', max_features=5000)    # default {max_df=1.0(float) so its proportion of word contain in all documents,
# simply if a words contain in all the document it is 1.0}, {min_df=1(int) so iteger directly denotes count of words in a document shows atleast 1 word contains in a documents then only it is consider}
tfidf_vect.fit(Data['consumer_complaint_narrative'])
print(tfidf_vect.stop_words)
print(tfidf_vect.vocabulary_)
xtrain_tfidf = tfidf_vect.transform(train_x)
#print(xtrain_tfidf)
xvalid_tfidf = tfidf_vect.transform(valid_x)

encoder = preprocessing.LabelEncoder()

# Train and fit the encoder

# Save the encoder object to a file named 'encoder.pkl'
joblib.dump(encoder, 'encoder.pkl')
# In[23]:


help(TfidfVectorizer)


# ## Model building
# 
# Suppose we are building a linear classifier on word-level TF-IDF vectors. We are using default hyper parameters for the classifier.

# In[24]:


print(xtrain_tfidf[0])


# In[25]:


model = linear_model.LogisticRegression().fit(xtrain_tfidf, train_y)


# In[26]:


model


# ## Model Evaluation

# In[27]:


def accuracy():
    # checking accuracy
    accuracy = metrics.accuracy_score(model.predict(xvalid_tfidf),valid_y)
    print("Accuracy: ",accuracy)
    print(metrics.classification_report(valid_y,model.predict(xvalid_tfidf), target_names=Data['product'].unique()))

accuracy()


# Examine these results:
# 1. The overall accuracy is 0.846. However as just observed this is as somewhat misleading beacuse of some cases like `money transfer`, `payday loan`, and `other financial service` are little misclassified.
# 3. The class imbalance is confirmed. Of the 33, 190, 210 cases are very less than as compaire to 4370 or 3700. 
# 4. The precision, recall and F1 all show that Debt collection, Mortgage, Credit card, Credit reporting, Bank account or service, Prepaid card cases are classified reasonably well, but the Consumer Loan, Student loan, Payday loan, Money transfers, Other financial service cases are not. As if those categories are misclassified is will directly cost to bank, beacause after going complaint to unexpected team they will not able to resolve the issues and ticket is going to send to appropriate team so, increases time range and directly impacts to customer satisfaction leads to high risk on customer relationship mangement by bank. since bank fully depends on customers they should not tollerate it.    

# Let's Analyse the correctly classified and misclassified data closely by heatmap structure and confusion matrix.

# In[28]:


def heat_conf():
    # confusion matrix
    conf_mat = metrics.confusion_matrix(valid_y,model.predict(xvalid_tfidf))
    print(conf_mat)
    # visualizing confusion matrix
    #category_id_df = Data[['product','category_id']].drop_duplicates().sort_values('category_id')
    #category_id_df
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(conf_mat, annot=True,fmt='d',cmap='BuPu',xticklabels=Data['product'].unique(),yticklabels=Data['product'].unique())
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
heat_conf()


# Examine the result:      
# The confusion matrix shows the following characteristics; a) Debt collection, Mortgage, Credit card, Consumer Loan, Credit reporting, Bank account or service, Prepaid card cases are classified reasonably well, b) but the Student loan, Payday loan, Money transfers, Other financial service cases are not however,

# In[29]:


category_id_df = pd.DataFrame()
category_id_df['category'] = Data['product'].unique()
category_id_df['category_id'] = category_id_df.index.values

#print(category_id_df.index.values)
category_to_id = dict(category_id_df[['category_id','category']].values)
category_to_id


# In[30]:


probabilities = model.predict_proba(xvalid_tfidf)
print(probabilities[:15,:])


# The above shows probablitic approach towards the classification problem, where the highest prob belongs to the perticular class. for ex: see the 1st row it belongs to class 3. its value ranges [0-1]
# 
# Now the below code shows that The desion_function() tells us on which side of the hyperplane generated by the classifier we are (and how far we are away from it). Based on that information, the estimator then label the examples with the corresponding label.

# In[31]:


y_score = model.decision_function(xvalid_tfidf)
print(y_score[:15,:])
predictions = model.predict(xvalid_tfidf)
print(predictions[:15])


# In[32]:


# Binarize the output
y = preprocessing.label_binarize(valid_y, classes=[0,1,2,3,4,5,6,7,8,9,10])
n_classes = y.shape[1]
y


# Finally, the code in the cell below computes and displays the ROC curve and AUC. The `roc_curve` and `auc` functions from the scikit-learn `metrics` package are used to compute these values. 

# In[33]:


def plot_auc(labels, colours=['orange']):
    
    # Compute ROC curve and ROC area for each class
    n_classes = len(category_to_id)
    # print(n_classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    ## Compute the false positive rate, true positive rate
    ## and threshold along with the AUC

    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y[:,i], y_score[:, i])
    #     print(fpr[i], tpr[i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10,6))
    for i, color in zip(range(n_classes), colours):
        plt.plot(fpr[i], tpr[i], color = color, label = 'AUC of class {0} = {1:0.2f}'.format(i,roc_auc[i]))
        
    ## Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
 
colours = ['aqua', 'darkorange', 'cornflowerblue','r','indigo','darkmagenta','green','brown','olive','darkcyan','violet']
plot_auc(y, colours)    # in place of probabilities we can put y_score of decision function as well. will get a same result.


# The above graph if for all the classified ROC curves with different colors. legends shows that all the auc are much higher and well correctly classified consumer complaints, mostly values are above 0.95 excepting some for 0.89. This curves shows that we have build a perfectly nice model as expected which is mostly classified correct complaints.

# ## Compute a weighted model

# Recall that a falsely classifying a some categorical product customer as different category, cost high to bank and followup time to customers increses. Given this situation, the results of the first model are not that good. There are two reasons for this:
# 
# 1. The class imbalance in the label has biased the training of the model. 
# 2. Nothing has been done to weight the results toward correctly classifying perticular product category data.
# 
# One approach to these problems is to weight the classes when computing the logistic regression model. The code in the cell below adds a `class_weight` argument to the call to the `LogisticRegression` function. In this case weights are chosen as 5 for class 5 and 10 for class 7  but you can also give another combination. Execute this code
# 
# **Note:** default class_weights are 1 

# In[34]:


# result_metric = metrics.precision_recall_fscore_support(train_y,model.predict(xtrain_tfidf))
# num_cases = result_metric[3][:]
# print(num_cases.sort())
# print(sum(num_cases))
# label_dic = {i:j for i,j in zip(num_cases,range(len(num_cases)))}
# print(label_dic)
# perc = []
# for i in range(len(num_cases)):
#     perc.append(num_cases[i]/sum(num_cases))
# perc_rev = sorted(perc,reverse=True)
# num_cases.sort()
# #print(perc_rev)
# class_weights = {label_dic[i]:j for i,j in zip(num_cases,perc_rev)}

# print(class_weights)


# In[35]:


from collections import Counter
Counter(train_y)


# In[36]:


model = linear_model.LogisticRegression(class_weight = {8:3.0, 9:3, 5:3.0, 7:20}).fit(xtrain_tfidf, train_y)
model


# In[37]:


accuracy()
heat_conf()


# In[38]:


y_score = model.decision_function(xvalid_tfidf)
print(y_score[:15,:])
predictions = model.predict(xvalid_tfidf)
print(predictions[:15])


# In[39]:


plot_auc(y, colours)    # in place of probabilities we can put y_score of decision function as well. will get a same result.


# The accuracy is slightly changed with respect to the unweighted model. The change could be more had we give more weights to `Payday Loan`, `Money Transfers`, `Other financial services` of the class than what we did here. The precision, recall and F1 are slightly better for the these cases. Reweighting the labels has moved the results in the desired direction, at least a bit.
# 
# The trade-off between true positive and false positive is similar to the unweighted model. 

# In[40]:


texts = ['This account popped up on my credit and it is not mines. I have filled out all the correct docs to show that i am victim of identity thief and will attach the ftc report with this complaint. Please block and remove this from my credit please XXXX XXXX XXXX Account Number: XXXX XXXX / 2019']
text_features = tfidf_vect.transform(texts)
predictions = model.predict(text_features)
# print(predictions)
print(texts)
print("   -Predicted as: {}".format(category_to_id[predictions[0]]))


# ## Cross validate model
# 
# To compute a better estimate of model performance, you can perform simple cross validation. The code in the cell performs the following processing:
# 1. Create a list of the metrics to be computed for each fold. 
# 2. Defines a logistic regression model object.
# 3. A 10 fold cross validation is performed using the `cross_validate` function from the scikit-learn `model_selection` package.
# 
# Execute this code. 

# In[41]:


#Labels = Labels.reshape(Labels.shape[0])
# Binarize the output
#Labels = preprocessing.label_binarize(Labels, classes=[0,1,2,3,4,5,6,7,8,9,10])
#Labels = np.array(Labels)
print(Labels1)


scoring = ['precision_macro','recall_macro']
logistic_mod = linear_model.LogisticRegression(C = 1.0, class_weight = {8:3.0, 9:3, 5:3.0, 7:20}) 
scores = ms.cross_validate(logistic_mod, Features, Labels1, scoring=scoring,
                        cv=10, return_train_score=False)


# The code in the cell below displays the performance metrics along with the mean and standard deviation, computed for each fold to the cross validation. The 'macro' versions of precision and recall are used. These macro versions average over the all the multinomial cases. 

# In[42]:


def print_format(f,x,y):
    print('Fold %2d    %4.3f        %4.3f' % (f, x, y))

def print_cv(scores):
    fold = [x + 1 for x in range(len(scores['test_precision_macro']))]
    print('         Precision     Recall')
    [print_format(f,x,y) for f,x,y in zip(fold, scores['test_precision_macro'], 
                                          scores['test_recall_macro'])]
                                              
    print('-' * 30)
    print('Mean       %4.3f        %4.3f '%
          (np.mean(scores['test_precision_macro']), np.mean(scores['test_recall_macro'])))  
    print('Std        %4.3f        %4.3f '%
          (np.std(scores['test_precision_macro']), np.std(scores['test_recall_macro'])))

print_cv(scores)  


# Notice that there is considerable variability in each of the performance metrics from fold to fold. Even so, the standard deviations are at least an order of magnitude than the means. It is clear that **any one fold does not provide a representative value of the performance metrics**. The later is a key point as to why cross validation is important when evaluating a machine learning model.  
# 
# Compare the performance metric values to the values obtained for the baseline model you created above. In general the metrics obtained by cross validation are lower. However, the metrics obtained for the baseline model are within 1 standard deviation of the average metrics from cross validation. 

# ## Optimize hyperparameters with nested cross validation
# 
# Given the variability observed in cross validation, it should be clear that performing model selection from a single training and evaluation is an uncertain proposition at best. Fortunately, the nested cross validation approach provides a better way to perform model selection. However, there is no guarantee that a model selection process will, in fact, improve a model. In some cases, it may prove to be that model selection has minimal impact. 
# 
# To start the nested cross validation process it is necessary to define the randomly sampled folds for the inner and outer loops. The code in the cell below uses the `KFolds` function from the scikit-learn `model_selection` package to define fold selection objects. Notice that the `shuffle = True` argument is used in both cases. This argument specifies that a random shuffle is preformed before folds are created, ensuring that the sampling of the folds for the inside and outside loops are independent. Notice that by creating these independent fold objects there is no need to actually create nested loops for this process. 
# 
# Execute this code.

# In[43]:


nr.seed(456)
inside = ms.KFold(n_splits=10, shuffle = True)
nr.seed(645)
outside = ms.KFold(n_splits=10, shuffle = True)


# An important decision in model selection searches is the choice of performance metric used to find the best model. For classification problems scikit-learn uses accuracy as the default metric. However, as you have seen previously, accuracy is not necessarily the best metric, particularly when there is a class imbalance as is the case here. There are a number of alternatives which one could choose for such a situation. In this case AUC will be used. 
# 
# The code below uses the `inside` k-fold object to execute the inside loop of the nested cross validation. Specifically, the steps are:
# 1. Define a dictionary with the grid of parameter values to search over. In this case there is only one parameter, `C`, with a list of values to try. In a more general case, the dictionary can contain values from multiple parameters, creating a multi-dimensional grid that the cross validation process will iterate over. In this case there are 5 hyperparameter values in the grid and 10-fold cross validation is being used. Thus, the model will be trained and evaluated 50 times. 
# 2. The logistic regression model object is defined. 
# 3. The cross validation search over the parameter grid is performed using the `GridSearch` function from the scikit-learn `model_selection` package. Notice that the cross validation folds are computed using the `inside` k-fold object.
# 
# 
# ****
# **Note:** Somewhat confusingly, the scikit-learn `LogisticRegression` function uses a regularization parameter `C` which is the inverse of the usual l2 regularization parameter $\lambda$. Thus, the smaller the parameter the stronger the regulation 
# ****
# 
# Execute this code.

# In[44]:


nr.seed(3456)
## Define the dictionary for the grid search and the model object to search on
param_grid = {"estimator__C": [0.1, 1, 10, 100, 1000]}
## Define the logistic regression model
logistic_mod = linear_model.LogisticRegression() 

clf_log = multiclass.OneVsRestClassifier(logistic_mod)
## Perform the grid search over the parameters
clf = ms.GridSearchCV(estimator = clf_log, param_grid = param_grid, 
                      cv = inside, # Use the inside folds
                      scoring = 'roc_auc',
                      return_train_score = True)


# As expected, there is considerable variation in AUC across the folds. However, all of these values are within 1 standard deviation of each other, and thus these differences cannot be considered significant. 

# In[45]:


model = linear_model.LogisticRegression(C=10, class_weight = {8:3.0, 9:3, 5:3.0, 7:20}).fit(xtrain_tfidf, train_y)
accuracy()
heat_conf()
y_score = model.decision_function(xvalid_tfidf)
# print(y_score[:15,:])
predictions = model.predict(xvalid_tfidf)
# print(predictions[:15])
    
plot_auc(y,colours) 


filename = 'ConsumerComplaintClassification.sav'

# Use the model to make predictions
joblib.dump(model, filename)



# In[46]:


# import pickle
# # open a file where you want to store the data
# file = open('customer_classification_model_lr.pkl','wb')
# # Dump information to that file
# pickle.dump(model, file)


# ## Support Vector machine model

# Nested cross validation is used to estimate the optimal hyperparameters and perform model selection for the nonlinear SVM model. 5 fold cross validation is used since training SVMs are computationally intensive to train. Additional folds would give better estimates but at the cost of greater computation time. Execute the code in the cell below to define inside and outside fold objects. 

# In[47]:


nr.seed(248)
inside = ms.KFold(n_splits=5, shuffle = True)
nr.seed(135)
outside = ms.KFold(n_splits=5, shuffle = True)


# In[48]:


Labels = preprocessing.label_binarize(Labels1, classes=[0,1,2,3,4,5,6,7,8,9,10])
print(Labels)

# The code in the cell below estimates the best hyperparameters using 5 fold cross validation. There are two points to notice here
# 1. In this case, a grid of two hyperparameters: C is the inverse of lambda of l2 regularization, and kernel Specifies the kernel type to be used in the algorithm. 
# 2. Since there is a class imbalance and a difference in the cost to the bank of misclassification of a customer complaints, class weights are used. 
# 3. The model is fit on the grid and the best estimated hyperparameters are printed. 
# In[49]:


# nr.seed(3456)
# ## Define the dictionary for the grid search and the model object to search on
# param_grid = {"estimator__C": [1, 10, 100, 1000]}
# ## Define the SVM model
# svc_clf = svm.LinearSVC() 

# svc_multi_clf = multiclass.OneVsRestClassifier(svc_clf)
# ## Perform the grid search over the parameters
# clf = ms.GridSearchCV(estimator = svc_multi_clf, param_grid = param_grid, 
#                       cv = inside, # Use the inside folds
#                       scoring = 'roc_auc',
#                       return_train_score = True)
# clf.fit(Features, Labels)
# print(clf.best_params_)


# In[50]:


# plot_cv(clf, param_grid)    


# In[51]:


# nr.seed(498)
# cv_estimate = cross_val_score(clf, Features, Labels, 
#                                  cv = outside) # Use the outside folds
# print('Mean performance metric = %4.3f' % np.mean(cv_estimate))

# print('SDT of the metric       = %4.3f' % np.std(cv_estimate))
# print('Outcomes by cv fold')
# for i, x in enumerate(cv_estimate):
#     print('Fold %2d    %4.3f' % (i+1, x))


# In[52]:


model = svm.LinearSVC(class_weight={8:3.0, 9:3, 5:3.0, 7:20}).fit(xtrain_tfidf, train_y)


# In[53]:


accuracy()
heat_conf()
y_score = model.decision_function(xvalid_tfidf)
# print(y_score[:15,:])
predictions = model.predict(xvalid_tfidf)
# print(predictions[:15])
    
plot_auc(y,colours) 


# In[54]:


# import pickle
# # open a file where you want to store the data
# file = open('customer_classification_model_svm.pkl','wb')
# # Dump information to that file
# pickle.dump(model, file)

