__author__ = 'Aaron J Masino'

import os
import numpy as np

import sepsis.cross_validate as scv
import sepsis.log_worker as slog
from sepsis.model_analsyis import nested_cross_validation_analysis as ncv_analysis

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

# ************************** GLOBAL PARAMETERS ******************************************
data_dir = os.path.join('..','data')
input_data_file = os.path.join(data_dir, "processed",
                               "PREPROCESSED_DATA_FILE_PATH.csv")
output_dir = os.path.join(data_dir, 'interim')
figure_dir = output_dir
metrics_output_file = os.path.join(output_dir, 'scoring_metrics.txt')
store_prediction_probs = True
prediction_prob_dir = os.path.join(output_dir, "prediction_probabilities")
selected_features_file = os.path.join(output_dir, "selected_features.txt")
seed = 287462
num_folds = 10  # number of stratified folds for outside loop
shuffle_on_split = False  # shuffle data before splitting into folds
# number of features for feature selection, recommended number positive examples in training/ 10
# if set to -1 use all features
num_features = -1

slog.log_items("Global Parameters\n", metrics_output_file,
               num_folds=num_folds,
               num_features=num_features,
               shuffle_on_split=shuffle_on_split,
               seed=seed)

# ************************* GLOBAL PARAMETERS END *************************************

# import data
all_data = pd.read_csv(input_data_file).sample(frac=1, random_state=seed)
y = all_data['sepsis']
X = all_data.drop('sepsis', axis=1).drop('sepsis_group', axis=1)

# Use stratified K-fold to get data split indices
skf = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=shuffle_on_split)
folds = {}
fold_idx = 0
for train_split, test_split in skf.split(X,y):
    folds[fold_idx] = test_split
    fold_idx += 1

# evaluate models
kn = num_features
if num_features == -1:
    kn = len(X.columns)

# wrapper method to enable passing a random state for scoring method of feature selector
def seeded_mutual_info_classif(X, y):
    return mutual_info_classif(X,y, random_state=seed)

feature_selector = SelectKBest(seeded_mutual_info_classif, k=kn)

if store_prediction_probs and not os.path.exists(prediction_prob_dir):
    os.makedirs(prediction_prob_dir)

# ************************* Model evaluations ****************************************
# --------------------------   logistic regression --------------------------------------------
parameter_candidates = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

model = LogisticRegression(penalty='l2',class_weight='balanced', random_state=seed)
ncv_analysis(model, parameter_candidates, folds, X, y, feature_selector, "Logistic Regression", "LR",
             figure_dir, metrics_output_file, store_prediction_probs,
             os.path.join(prediction_prob_dir, "LR_pred_probs.csv"),
             os.path.join(prediction_prob_dir, "LR_targets.csv"), seed=seed,
             selected_features_file=selected_features_file,
             feature_coefs_file=os.path.join(output_dir,"LR_coefs.csv")
             )

#--------------------------  support vector machine ----------------------------------------------
parameter_candidates = [{'C': [0.01, 0.1, 1, 10, 100],
                         'gamma': [0.01, 0.1, 1, 10, 100]}]
model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=seed)
ncv_analysis(model, parameter_candidates, folds, X, y, feature_selector, "SVM (RBF)", "SVM-RBF",
             figure_dir, metrics_output_file, store_prediction_probs,
             os.path.join(prediction_prob_dir, "SVM_pred_probs.csv"),
             os.path.join(prediction_prob_dir, "SVM_targets.csv"), seed=seed, n_jobs=4)

#  --------------------------  Gaussian NB -------------------------------------------------------
model = GaussianNB()
metric_values, fpr_scores, tpr_scores, precision_scores, recall_scores = \
   scv.nested_cross_validate(model, None, folds, X, y, feature_selector=feature_selector)
ncv_analysis(model, None, folds, X, y, feature_selector, "Naive Bayes", "NaiveBayes",
             figure_dir, metrics_output_file, store_prediction_probs,
             os.path.join(prediction_prob_dir, "NaiveBayes_pred_probs.csv"),
             os.path.join(prediction_prob_dir, "NaiveBayes_targets.csv"), seed=seed)

#  --------------------------  Gaussian Process -------------------------------------------
model = GaussianProcessClassifier(random_state = seed)
metric_values, fpr_scores, tpr_scores, precision_scores, recall_scores = \
    scv.nested_cross_validate(model, None, folds, X, y, feature_selector=feature_selector)
ncv_analysis(model, None, folds, X, y, feature_selector, "Gaussian Process", "GaussianProcess",
             figure_dir, metrics_output_file, store_prediction_probs,
             os.path.join(prediction_prob_dir, "GaussianProcess_pred_probs.csv"),
             os.path.join(prediction_prob_dir, "GaussianProcess_targets.csv"), seed=seed)

#  --------------------------  Random Forest --------------------------------------------
parameter_candidates = [{'n_estimators': [10, 50, 100, 200],
                         'criterion': ['gini','entropy'],
                         'max_depth': [2, 4, 6]}]
model = RandomForestClassifier(random_state=seed, class_weight='balanced')
ncv_analysis(model, parameter_candidates, folds, X, y, feature_selector, "Random Forest", "RandomForest",
             figure_dir, metrics_output_file, store_prediction_probs,
             os.path.join(prediction_prob_dir, "RandomForest_pred_probs.csv"),
             os.path.join(prediction_prob_dir, "RandomForest_targets.csv"), seed=seed)

#  --------------------------  AdaBoost -------------------------------------------------
parameter_candidates = [{'base_estimator': [DecisionTreeClassifier(),
                                            LogisticRegression(class_weight='balanced', random_state=seed),
                                            SVC(kernel='rbf', probability=True,class_weight='balanced', random_state=seed)],
                         'n_estimators': [50, 100],
                         'learning_rate': [1.0, 0.5, 0.1]}]
model = AdaBoostClassifier(random_state=seed)
ncv_analysis(model, parameter_candidates, folds, X, y, feature_selector, "AdaBoost", "AdaBoost",
             figure_dir, metrics_output_file, store_prediction_probs,
             os.path.join(prediction_prob_dir, "AdaBoost_pred_probs.csv"),
             os.path.join(prediction_prob_dir, "AdaBoost_targets.csv"), seed=seed)


#  --------------------------  KNN Classifier ---------------------------------------
parameter_candidates = [{'n_neighbors': [5, 10],
                         'weights': ['uniform', 'distance']}]
model = KNeighborsClassifier()
ncv_analysis(model, parameter_candidates, folds, X, y, feature_selector, "KNN", "KNN",
             figure_dir, metrics_output_file, store_prediction_probs,
             os.path.join(prediction_prob_dir, "KNN_pred_probs.csv"),
             os.path.join(prediction_prob_dir, "KNN_targets.csv"), seed=seed)

# --------------------------  Gradient Boosting -------------------------------------------
# wrapper class to fix sklearn bug
class init:
    def __init__(self, est):
        self.est = est
    def predict(self, X):
        return self.est.predict_proba(X)[:,1][:,np.newaxis]
    def fit(self, X, y, *kwarg):
        self.est.fit(X, y)


model = GradientBoostingClassifier(random_state=seed)
parameter_candidates = [{'n_estimators': [50, 100, 200],
                         'max_depth': [3, 5, 10],
                         # 'init': [None,
                         #          init(DecisionTreeClassifier(class_weight='balanced')),
                         #          init(LogisticRegression(class_weight='balanced', random_state=seed)),
                         #          init(SVC(kernel='rbf', probability=True,class_weight='balanced'))]
                         }
                        ]
ncv_analysis(model, parameter_candidates, folds, X, y, feature_selector, "Gradient Boost", "GradBoost",
             figure_dir, metrics_output_file, store_prediction_probs,
             os.path.join(prediction_prob_dir, "GradBoost_pred_probs.csv"),
             os.path.join(prediction_prob_dir, "GradBoost_targets.csv"), seed=seed)
