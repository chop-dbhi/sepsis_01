__author__ = 'Aaron J Masino'

import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import learning_curve
from sklearn.neighbors.kde import KernelDensity

LABEL_FONT_SIZE = 16
TITLE_FONT_SIZE = 18
LEGEND_FONT_SIZE = 16
TICK_FONT_SIZE = 14
MULTI_FIG_SIZE = (16, 14)
SINGLE_FIG_SIZE = (8,6)
MARKER_SIZE = 10

def overlayHists(df1, df2, legend = None):
    num_figs = len(df1.columns)
    nrows = math.ceil(num_figs/3.0)
    ncols = 3
    fig, axarr= plt.subplots(nrows, ncols, figsize=MULTI_FIG_SIZE)
    cidx = 0
    for r in range(nrows):
        for c in range(ncols):
            if cidx < num_figs:
                h1_vals, h1_bins = np.histogram(df1[df1.columns[cidx]],normed=True)
                h2_vals, h2_bins = np.histogram(df2[df2.columns[cidx]],normed=True)
                width = (h1_bins[1] - h1_bins[0]) / 3.0
                ax = axarr[r,c]
                ax.bar(h1_bins[:-1], h1_vals, width=width, facecolor='blue')
                ax.bar(h2_bins[:-1]+width, h2_vals, width=width, facecolor='red')
                ax.set_title(df1.columns[cidx].replace("_"," "))
                ax.legend(legend)

            cidx += 1
    if legend is not None:
        plt.legend(legend)
    return (fig, axarr)


def overlayKDEs(df1, df2, legend, titles, figsize=MULTI_FIG_SIZE, num_points=None):
    num_figs = len(df1.columns)
    nrows = math.ceil(num_figs / 2.0)
    ncols = 2
    fig, axarr = plt.subplots(nrows, ncols, figsize=figsize)
    # fig.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    cidx = 0
    for r in range(nrows):
        for c in range(ncols):
            if cidx < num_figs:
                d1 = df1[df1.columns[cidx]]
                d2 = df2[df2.columns[cidx]]
                start = np.min((np.floor(0.9 * d1.min()), np.floor(0.9 * d1.min())))
                stop = np.max((np.ceil(1.1 * d1.max()), np.ceil(1.1 * d2.max())))
                if num_points is not None:
                    N = num_points
                else:
                    N = 5 * len(d1)
                X_plot = np.linspace(start, stop, N)[:, np.newaxis]
                X1 = np.array(d1)[:, np.newaxis]
                X2 = np.array(d2)[:, np.newaxis]
                # kde1 = KernelDensity(kernel='epanechnikov', bandwidth=0.5).fit(X1)
                kde1 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X1)
                log_dens1 = kde1.score_samples(X_plot)
                kde2 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X2)
                # kde2 = KernelDensity(kernel='epanechnikov', bandwidth=0.5).fit(X2)
                log_dens2 = kde2.score_samples(X_plot)
                ax = axarr[r, c]
                # ax.grid(color='whitesmoke')
                ax.plot(X_plot[:, 0], np.exp(log_dens1), '--', color='dimgrey', lw=2)
                ax.plot(X_plot[:, 0], np.exp(log_dens2), '-', color='black', lw=2)
                # ax.bar(h1_bins[:-1], h1_vals, width=width, facecolor='lightgrey')
                # ax.bar(h2_bins[:-1]+width, h2_vals, width=width, facecolor='dimgrey')
                ax.set_title(titles[cidx], fontsize=TITLE_FONT_SIZE)
                set_tick_fontsize(ax, TICK_FONT_SIZE)
                if cidx == 0:
                    ax.legend(legend, fontsize=LEGEND_FONT_SIZE)

            cidx += 1
    return fig, axarr

def overlayScatters(df1, df2, legend):
    num_figs = len(df1.columns)
    fig, axarr = plt.subplots(num_figs, num_figs, figsize=MULTI_FIG_SIZE)
    for r in range(num_figs):
        for c in range(num_figs):
            ax = axarr[r, c]

            if r == c:
                # plot histogram
                h1_vals, h1_bins = np.histogram(df1[df1.columns[r]], normed=True)
                h2_vals, h2_bins = np.histogram(df2[df2.columns[r]], normed=True)
                width = (h1_bins[1] - h1_bins[0]) / 3.0
                ax.bar(h1_bins[:-1], h1_vals, width=width, facecolor='blue')
                ax.bar(h2_bins[:-1] + width, h2_vals, width=width, facecolor='red')
                ax.set_title(df1.columns[r].replace("_", " "))
                ax.legend(legend)
            else:
                x1 = df1[df1.columns[c]]
                y1 = df1[df1.columns[r]]
                x2 = df2[df2.columns[c]]
                y2 = df2[df2.columns[r]]
                ax.scatter(x1, y1, color="blue")
                ax.scatter(x2, y2, color="red", alpha=0.2)
                ax.legend(legend)
            if c == 0:
                ax.set_ylabel(df1.columns[r], fontsize=16)
            if r == num_figs - 1:
                ax.set_xlabel(df1.columns[c], fontsize=16)
            ax.set_title('')
    return (fig, axarr)

def plot_learning_curves(model, X, y, title, cv = 10, bins=10,
                         n_jobs=1, metric='roc_auc', metric_pretty_print = 'AU-ROC',
                         figsize=SINGLE_FIG_SIZE):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    lw = 2
    train_sizes = np.linspace(1/float(bins), 1.0, bins)

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=metric)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="k")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="k")
    plt.plot(train_sizes, train_scores_mean, 's--', color="k",
             label="Training score", lw=lw, markersize=MARKER_SIZE)
    plt.plot(train_sizes, test_scores_mean, 'o-', color="k",
             label="Validation score", lw=lw, markersize=MARKER_SIZE)

    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel('Training Set Size', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('{0}'.format(metric_pretty_print), fontsize=LABEL_FONT_SIZE)
    set_tick_fontsize(ax, TICK_FONT_SIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    ax.grid()
    return fig, ax

def plot_nested_cv_ROC(fpr_scores, tpr_scores, auc_scores, title):
    fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIG_SIZE)
    lw = 2

    # plot min auc
    idx = (np.abs(auc_scores - np.min(auc_scores))).argmin()
    fpr = fpr_scores[idx]
    tpr = tpr_scores[idx]
    ax.plot(fpr, tpr, '--o', color='grey', markersize=MARKER_SIZE,
            lw=lw, label='Minimum AUC = %0.2f' % auc_scores[idx])

    # plot auc that is closest to median
    idx = (np.abs(auc_scores - np.median(auc_scores))).argmin()
    fpr = fpr_scores[idx]
    tpr = tpr_scores[idx]
    ax.plot(fpr, tpr, color='black',markersize=MARKER_SIZE,
             lw=lw+1, label='Median AUC = %0.2f' % auc_scores[idx])


    # plot max auc
    idx = (np.abs(auc_scores - np.max(auc_scores))).argmin()
    fpr = fpr_scores[idx]
    tpr = tpr_scores[idx]
    ax.plot(fpr, tpr, '--s', color='grey',markersize=MARKER_SIZE,
            lw=lw, label='Maximum AUC = %0.2f' % auc_scores[idx])

    ax.plot([0, 1], [0, 1], color='darkgrey', lw=lw, linestyle='--', label='Random Guess')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('1 - Specificity', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Sensitivity', fontsize=LABEL_FONT_SIZE)
    set_tick_fontsize(ax, TICK_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE)
    ax.grid()
    return fig, ax

def plot_nested_cv_PR(precision_scores, recall_scores, avg_precision_scores, title):
    fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIG_SIZE)
    lw = 2

    # plot min pr
    idx = (np.abs(avg_precision_scores-np.min(avg_precision_scores))).argmin()
    ps = precision_scores[idx]
    rs = recall_scores[idx]
    ax.plot(rs, ps, '--o', color='darkgrey', lw = lw,
            label = 'Minimum Avg Precision = %0.2f' % avg_precision_scores[idx])

    # plot median
    idx = (np.abs(avg_precision_scores - np.median(avg_precision_scores))).argmin()
    ps = precision_scores[idx]
    rs = recall_scores[idx]
    ax.step(rs, ps, color='black', alpha=0.5, where='post', lw=lw+1,
            label='Median Avg Precision = %0.2f' % avg_precision_scores[idx])
    ax.fill_between(rs, ps, step='post', alpha=0.1, color='black')

    # plot max pr
    idx = (np.abs(avg_precision_scores - np.max(avg_precision_scores))).argmin()
    ps = precision_scores[idx]
    rs = recall_scores[idx]
    ax.plot(rs, ps, '--s', color='darkgrey', lw=lw,
            label='Maximum Avg Precision = %0.2f' % avg_precision_scores[idx])

    ax.set_xlabel('Sensitivity', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Precision', fontsize=LABEL_FONT_SIZE)
    set_tick_fontsize(ax, TICK_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
    ax.grid()
    return fig, ax


def set_tick_fontsize(ax, fs):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fs)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)


def savePdf(filename,dpi=300, close_fig_when_done=True):
    pdf = PdfPages(filename)
    pdf.savefig(dpi=dpi, bbox_inches='tight', pad_inches=.15)
    pdf.close()
