__author__ = 'Aaron J Masino'

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import metrics
from sepsis import log_worker as slog

# METRICS KEYS
ACCURACY = 'accuracy'
F1 = 'f1'
SENSITIVITY = 'sensitivity'
SPECIFICITY = 'specificity'
PRECISION = 'precision'
ROC_AUC = 'ROC-AUC'
AVERAGE_PRECISION = 'avg_precision'
NPV = 'npv'
METRICS_KEYS = [ACCURACY, F1, SENSITIVITY, SPECIFICITY, PRECISION, ROC_AUC, AVERAGE_PRECISION, NPV]

def nested_cross_validate(model,
                          parameter_candidates,
                          folds,
                          X,
                          y,
                          cv_scoring='roc_auc',
                          feature_selector=None,
                          print_progress=True,
                          store_prediction_probabilities=False,
                          prob_file=None,
                          target_file=None,
                          seed=123456,
                          n_jobs=1,
                          selected_features_file=None,
                          feature_coefs_file=None):
    num_folds = len(folds.keys())

    metrics_scores = {}
    fpr_scores = []
    tpr_scores = []
    precision_scores = []
    recall_scores = []

    for k in METRICS_KEYS:
        metrics_scores[k] = np.zeros(num_folds)

    if selected_features_file is not None:
        feature_counts = {}


    if feature_coefs_file is not None:
        feature_coefs = {}

    # train, eval runs
    for fold_idx, test_indices in folds.items():
        if print_progress:
            print("Starting run {0} of {1} for {2}".format(fold_idx, len(folds), str(model).split('(')[0]))
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        cv_indices = None
        for idx in range(num_folds):
            if idx != fold_idx:
                if cv_indices is None:
                    cv_indices = folds[idx]
                else:
                    cv_indices = np.concatenate((cv_indices, folds[idx]))
        X_cv = X.iloc[cv_indices]
        y_cv = y.iloc[cv_indices]

        # use sklearn grid search to identify best parameters for this model using
        # training folds
        if feature_selector is not None:
            feature_selector.fit(X_cv, y_cv)
            idx_selected = feature_selector.get_support(indices=True)
            if selected_features_file is not None:
                for cn in X_cv.columns[idx_selected]:
                    if cn in feature_counts:
                        feature_counts[cn] = feature_counts[cn] + 1
                    else:
                        feature_counts[cn] = 1
            if feature_coefs_file is not None:
                feature_name_from_index = {}
                for idx,cn in enumerate(X_cv.columns[idx_selected]):
                    feature_name_from_index[idx] = cn

            X_cv = X_cv[X_cv.columns[idx_selected]]
            X_test = X_test[X_test.columns[idx_selected]]

        if parameter_candidates is None:
            best_model = model
            model.fit(X_cv, y_cv)
        else:
            if print_progress:
                print("\tStarting grid search ....")
            skfgs = StratifiedKFold(n_splits=num_folds-1, random_state=seed)
            gs = GridSearchCV(estimator=model, param_grid=parameter_candidates, cv=skfgs, scoring=cv_scoring,
                              n_jobs=n_jobs)
            gs.fit(X_cv, y_cv)
            if print_progress:
                print("\tGrid search complete ....")
            # train model on all training folds and evaluate on unseen test fold
            best_model = gs.best_estimator_
            best_model.fit(X_cv,y_cv)
            if feature_coefs_file is not None:
                for idx, c in enumerate(best_model.coef_[0]):
                    cn = feature_name_from_index[idx]
                    if cn in feature_coefs:
                        feature_coefs[cn].append(c)
                    else:
                        feature_coefs[cn]=[c]

        y_pred = best_model.predict(X_test)
        p_pred = best_model.predict_proba(X_test)
        p_pred = p_pred[:,1]
        if store_prediction_probabilities:
            l = ""
            for p in p_pred:
                l = "{0},{1}".format(l,p)
            l = l[1:]
            slog.log_line("{0}\n".format(l),prob_file)
        if store_prediction_probabilities:
            l = ""
            for v in y_test:
                l = "{0},{1}".format(l,v)
            l = l[1:]
            slog.log_line("{0}\n".format(l),target_file)
        metrics_scores[ACCURACY][fold_idx] = metrics.accuracy_score(y_test, y_pred)
        metrics_scores[F1][fold_idx] = metrics.f1_score(y_test, y_pred)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
        metrics_scores[SPECIFICITY][fold_idx] = float(tn) / float((tn+fp))
        metrics_scores[PRECISION][fold_idx] = metrics.precision_score(y_test, y_pred)
        metrics_scores[SENSITIVITY][fold_idx] = metrics.recall_score(y_test, y_pred)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, p_pred)
        metrics_scores[ROC_AUC][fold_idx] = metrics.auc(fpr, tpr)
        fpr_scores.append(fpr)
        tpr_scores.append(tpr)
        if tn+fn == 0:
            npv = 0
        else:
            npv = tn / float(tn+fn)
        metrics_scores[NPV][fold_idx] = npv

        metrics_scores[AVERAGE_PRECISION][fold_idx] = metrics.average_precision_score(y_test, p_pred)
        pscores, rscores, _ = metrics.precision_recall_curve(y_test, p_pred)
        precision_scores.append(pscores)
        recall_scores.append(rscores)

    if selected_features_file is not None:
        slog.log_dictionary(feature_counts, "feature,count\n", file=selected_features_file)

    if feature_coefs_file is not None:
        for k,lv in feature_coefs.items():
            line="{0}:".format(k)
            for v in lv:
                line = "{0}{1},".format(line,v)
            slog.log_line("{0}\n".format(line[0:-1]),feature_coefs_file)

    return metrics_scores, fpr_scores, tpr_scores, precision_scores, recall_scores
