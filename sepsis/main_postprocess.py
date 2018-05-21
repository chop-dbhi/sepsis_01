__author__ = 'Aaron J Masino'

import os
from sepsis import mathx
import sepsis.evaluation as evaluate
from sepsis import log_worker as slog
from sklearn import metrics
import numpy as np

# ************************** GLOBAL PARAMETERS ******************************************
data_dir = os.path.join('..','data')
input_data_dir = os.path.join(data_dir, "results",
                              "PATH/TO/PREDICTION/PROBABILITES/FILE",
                              "prediction_probabilities")


_prob_file = os.path.join(input_data_dir,"{0}_pred_probs.csv")
_targ_file = os.path.join(input_data_dir,"{0}_targets.csv")
file_prefixes = ['AdaBoost', 'GradBoost','GaussianProcess', 'KNN', 'LR', 'NaiveBayes', 'RandomForest', 'SVM']
target_metric_name = evaluate.SENSITIVITY
target_metric_value = 0.8
ci_level = 0.95
metrics_output_file = os.path.join(data_dir, 'interim', 'scoring_metrics_fixed_{0}_{1}.csv'.
                                   format(target_metric_name,target_metric_value))

metrics_ranges_output_file = os.path.join(data_dir, 'interim', 'scoring_metrics_ranges_fixed_{0}_{1}.csv'.
                                   format(target_metric_name,target_metric_value))
# ************************* GLOBAL PARAMETERS END *************************************
def loaddata(file):
    with open(file,'r') as f:
        all_data = []
        for line in f.readlines():
            data = [float(x) for x in line.split(",")]
            all_data.append(data)
        return all_data

line = "model,acc,acc_std,acc_cil,acc_cih,f1,f1_std,f1_cil,f1_cih,sensitivity,sensitivity_std,sensitivity_cil,sensitivity_cih," \
       "specificity,specificity_std,specificity_cil,specificity_cih,precision,precision_std,precision_cil,precision_cih," \
       "npv,npv_std,npv_cil,npv_cih\n"
slog.log_line(line, metrics_output_file)

range_line = "model,acc,acc_low,acc_high,f1,f1_low,f1_high,sensitivity,sensitivity_low,sensitivity_high," \
       "specificity,specificity_low,specificity_high,precision,precision_low,precision_high," \
       "npv,npv_low,npv_high\n"
slog.log_line(range_line, metrics_ranges_output_file)

for fp in file_prefixes:
    probs = loaddata(_prob_file.format(fp))
    targs = loaddata(_targ_file.format(fp))
    acc, f1, sen, spec, precis, npv = evaluate.compute_metrics(targs, probs, target_metric_value, target_metric_name)
    scores = [acc, f1, sen, spec, precis, npv]
    line = "{0},".format(fp)
    range_line = "{0},".format(fp)
    for score in scores:
        m, s, cil, cih = mathx.mean_confidence_interval(score, ci_level)
        line="{0}{1},{2},{3},{4},".format(line, m,s,cil,cih)
        low = np.min(score)
        high = np.max(score)
        range_line="{0}{1},{2},{3},".format(range_line, m, low, high)
    line = "{0}\n".format(line[0:-1])
    range_line="{0}\n".format(range_line[0:-1])
    slog.log_line(line, metrics_output_file)
    slog.log_line(range_line, metrics_ranges_output_file)





