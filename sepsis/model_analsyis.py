__author__ = 'Aaron J Masino'

import os
import sepsis.plotting as splt
import sepsis.cross_validate as scv
import sepsis.log_worker as slog
from matplotlib import pyplot as plt


def nested_cross_validation_analysis(model,
                                     parameter_candidates,
                                     folds,
                                     X,
                                     y,
                                     feature_selector,
                                     model_plot_label,
                                     model_file_label,
                                     figure_dir,
                                     metrics_output_file,
                                     store_prediction_probabilities=False,
                                     prob_file=None,
                                     target_file=None,
                                     seed=12345,
                                     n_jobs = 1,
                                     selected_features_file=None,
                                     feature_coefs_file=None):
    metric_values, fpr_scores, tpr_scores, precision_scores, recall_scores = \
        scv.nested_cross_validate(model, parameter_candidates, folds, X, y, feature_selector=feature_selector,
                                  store_prediction_probabilities=store_prediction_probabilities,
                                  prob_file=prob_file, target_file=target_file, seed=seed, n_jobs=n_jobs,
                                  selected_features_file=selected_features_file,
                                  feature_coefs_file=feature_coefs_file)
    slog.log_metrics(metric_values, model_plot_label, print_to_screen=True, file=metrics_output_file)
    fig, ax = splt.plot_nested_cv_ROC(fpr_scores, tpr_scores, metric_values[scv.ROC_AUC],
                                      "{0} ROC".format(model_plot_label))
    splt.savePdf(os.path.join(figure_dir, '{0}-ROC.pdf'.format(model_file_label)))
    plt.close(fig)
    fig, ax = splt.plot_nested_cv_PR(precision_scores, recall_scores, metric_values[scv.AVERAGE_PRECISION],
                                     "{0} PR Curve".format(model_plot_label))
    splt.savePdf(os.path.join(figure_dir, '{0}-PRcurve.pdf'.format(model_file_label)))
    plt.close(fig)
    fig, ax = splt.plot_learning_curves(model, X, y, "{0} Learning Curve".format(model_plot_label), metric='f1',
                                        metric_pretty_print='F1 Score')
    splt.savePdf(os.path.join(figure_dir, '{0}-LearningCurve.pdf'.format(model_file_label)))
    plt.close(fig)