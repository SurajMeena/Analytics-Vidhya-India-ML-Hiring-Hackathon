# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 22:22:03 2019

@author: Suraj
"""
#def modelfit(alg, dtrain, performCV=True, printFeatureImportance=True, cv_folds=5):
#    #Fit the algorithm on the data
#    alg.fit(X_train, y_train)
#        
#    #Predict training set:
#    dtrain_predictions = alg.predict(X_train)
#    dtrain_predprob = alg.predict_proba(X_train)[:,1]
#    
#    #Perform cross-validation:
#    if performCV:
#        cv_score = cross_validation.cross_val_score(alg, X_train, y_train, cv=cv_folds, scoring='f1_macro')
#    
#    #Print model report:
#    print ("\nModel Report")
#    print ("AUC Score (Train): %f" % metrics.f1_macro(y_train, dtrain_predprob))
#    
#    if performCV:
#        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
#        
#    #Print Feature Importance:
#    if printFeatureImportance:
#        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
#        feat_imp.plot(kind='bar', title='Feature Importances')
#        plt.ylabel('Feature Importance Score')
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import sklearn
import statistics
from xgboost import XGBClassifier

# Applying Stratified k-Fold Cross Validation
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score 
scv = StratifiedKFold(n_splits=10)
params_grid = [{'min_samples_leaf': [2, 5, 10, 15, 20], 'min_samples_split': [2, 3, 4, 5, 6, 7, 10, 100]
                    , 'max_depth': [1, 4, 8, 16 ,32]
                    , 'criterion': ['gini']}
                    , {'criterion': ['entropy'], 'min_samples_leaf': [2, 5, 10, 15, 20]
                    , 'min_samples_split' : [2, 3, 4, 5, 6, 7, 10, 100]
                    , 'max_depth': [1, 4, 8, 16 ,32]}]
#params_grid = [{'min_samples_leaf': [0.1, 0.5, 5], 'min_samples_split': [2, 3, 4, 5, 6, 7, 10]
#                    , 'max_depth': [1, 4, 8, 16 ,32]
#                    , 'criterion': ['gini']}
#                    , {'criterion': ['entropy'], 'min_samples_leaf': [0.1, 0.5, 5]
#                    , 'min_samples_split' : [0.1, 2, 3 , 4, 5, 6, 8, 10]
#                    , 'max_depth': [1, 4, 8, 16 ,32]}]
# Performing CV to tune parameters for best SVM fit
gridsearch = GridSearchCV(DecisionTreeClassifier(random_state=0, class_weight='balanced'
                                                 , presort=True), params_grid
                                                , cv= scv, scoring='f1_macro', n_jobs=-1, verbose=50)
gridsearch.fit(X_res, y_res)
gridsearch.best_params_
gridsearch.best_score_

oversampler = SMOTE(sampling_strategy=0.25, random_state=0, k_neighbors=10)
X_res, y_res = oversampler.fit_resample(X_train, y_train)

model = XGBClassifier()
model.fit(X_train, y_train)

X_res = pd.DataFrame(X_res)
y_res = pd.DataFrame(y_res)
df_all_rows = pd.concat([X_res, y_res], axis=1)
df_all_rows = df_all_rows.sample(frac=1).reset_index(drop=True)
X_res = df_all_rows.iloc[:,:-1].values
#y_res = df_all_rows[0]
#y_res = y_res.T.reset_index(drop=True).T
y_res = df_all_rows.iloc[:,-1].values

DT = DecisionTreeClassifier(random_state=0, class_weight='balanced', presort=True)
DT.fit(X_train, y_train)

svm_model = SVC(kernel='rbf', random_state=0, gamma='auto')
svm_model.fit(X_train, y_train)

NB = GaussianNB()
NB.fit(X_train, y_train)

RFC = RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=100 )
RFC.fit(X_res, y_res)

#predictors = [x for x in train.columns if x not in [target, IDcol]]
#param_test1 = {'n_estimators':range(20,81,10)}
#gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
#param_grid = param_test1, scoring='f1_macro',n_jobs=-1,iid=False, cv=10, verbose=30)
#gsearch1.fit(X_train,y_train)
#gsearch1.best_params_, gsearch1.best_score_ 

RUS = RandomUnderSampler(sampling_strategy=0.20, return_indices=True, replacement=True, random_state=0)
X_res, y_res, index = RUS.fit_resample(X_train, y_train)


#sorted(sklearn.metrics.SCORERS.keys())

crossvalscore = cross_val_score(estimator=model, X=X_train, y=y_train, cv=scv,  scoring = 'f1_macro', n_jobs=-1, verbose=30)
crossvalscore.mean()
crossvalscore.std()

# save the model to disk
#import pickle
#filename = 'GBC_parametrized_implementation.sav'
#pickle.dump(gsearch5.best_estimator_, open(filename, 'wb'))

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, y_test)

# Predicting on splitted test set and training set
import joblib
# Save the model as a pickle in a file
joblib.dump(gridsearch.best_estimator_, 'RandomUnderSampler.pkl')

# Load the model from the file
DT_from_joblib = joblib.load('Oversampler.pkl')
#DT_from_joblib2 = joblib.load('DT_parametrized2_withoutOrigination.pkl')
#DT_from_joblib3 = joblib.load('RandomUnderSampler.pkl')
#
#pred1=DT_from_joblib.predict(test)
#pred2=DT_from_joblib2.predict(test)
#pred3=DT_from_joblib3.predict(test)
#
#y_test_preds = np.array([])
#for i in range(0,len(test)):
#    y_test_preds = np.append(y_test_preds, statistics.mode([pred1[i], pred2[i], pred3[i]]))

y_pred_train = gridsearch.best_estimator_.predict(X_train)
y_pred = gridsearch.best_estimator_.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import f1_score
MacroF1_test = f1_score(y_test, y_pred, average='macro')
#MacroF1_train = f1_score(y_train, y_pred_train, average='macro')
# --classification report --
Report = metrics.classification_report(y_test, y_pred, labels=[0,1])
#Predicting on test dataframe
index = pd.DataFrame(test['loan_id'])
index.reset_index(drop=True, inplace=True)
test = test.drop(["loan_id"], axis=1)
# Use the loaded model to make predictions
y_test_preds = DT_from_joblib.predict(test)
#Preparing to write a CSV file

submission_format = pd.DataFrame(y_test_preds)
submission_format.columns = ['m13']

FinalSubmission = pd.concat([index, submission_format], sort=False, axis=1)
FinalSubmission = FinalSubmission.sort_values(by='loan_id')
FinalSubmission.to_csv(r"C:\Users\Suraj\Google Drive\Ubuntu\ML Problems\India ML Hiring Hackathon\VotedClassifier.csv", header=True, index=None)
 