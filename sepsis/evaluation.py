__author__ = 'Aaron J Masino'


# METRICS KEYS
SENSITIVITY = 'sensitivity'
SPECIFICITY = 'specificity'

from sklearn import metrics
import numpy as np

def predictions(predicted_prob, threshold):
    y_pred = np.zeros(len(predicted_prob))
    for idx, p in enumerate(predicted_prob):
        if p>threshold:
            y_pred[idx]=1
    return y_pred

def threshold_for(y_true, predicted_prob, metric_target_value, metric=SENSITIVITY):
    # fpr = 1 - specificity
    # tpr = sensitivity
    fpr, sensitivity, thresholds = metrics.roc_curve(y_true, predicted_prob)
    if metric == SENSITIVITY:
        return locate_decision_threshold(metric_target_value, sensitivity, thresholds)

    if metric == SPECIFICITY:
        specificity = 1 - fpr
        specificity.sort()
        return locate_decision_threshold(metric_target_value, specificity, reversed(thresholds))

def locate_decision_threshold(target_value, sorted_metric_values, prob_thresholds):
    for idx, t in enumerate(prob_thresholds):
        if sorted_metric_values[idx] >= target_value:
            return t

def binary_confusion(y_true, y_pred):
    tp = np.where((y_pred == y_true) & (y_pred == 1))
    fp = np.where(np.not_equal(y_pred, y_true) & (y_pred == 1))
    tn = np.where((y_pred == y_true) & (y_pred == 0))
    fn = np.where(np.not_equal(y_pred, y_true) & (y_pred == 0))
    return tp, fp, tn, fn

def compute_metrics(y_true, predicted_prob, metric_target_value, metric=SENSITIVITY):
    nrows = len(predicted_prob)
    sensitivity = np.zeros(nrows)
    specificity = np.zeros(nrows)
    precision = np.zeros(nrows)
    accuracy = np.zeros(nrows)
    f1 = np.zeros(nrows)
    npv = np.zeros(nrows)

    for idx in range(nrows):
        th = threshold_for(y_true[idx], predicted_prob[idx], metric_target_value, metric)
        y_pred = np.zeros(len(predicted_prob[idx]))
        y_pred[np.where(np.array(predicted_prob[idx]) >= th)] = 1
        tp, fp, tn, fn = binary_confusion(y_true[idx], y_pred)
        tp = tp[0].shape[0]
        fp = fp[0].shape[0]
        tn = tn[0].shape[0]
        fn = fn[0].shape[0]
        cp = float(tp + fn)
        cn = float(tn + fp)
        sensitivity[idx] = tp / cp
        specificity[idx] = tn / cn
        precision[idx] = tp / float(tp + fp)
        accuracy[idx] = (tp + tn)/(cp + cn)
        f1[idx] = precision[idx] * sensitivity[idx] / (precision[idx] + sensitivity[idx])
        if tn+fn ==0:
            npv[idx] = 0
        else:
            npv[idx] = tn/ float(tn + fn)
    return accuracy, f1, sensitivity, specificity, precision, npv
