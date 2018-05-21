__author__ = 'Aaron J Masino'

from sepsis import cross_validate as scv
import numpy as np


def log_metrics(metric_values, header, print_to_screen = False, file=None):
    if print_to_screen:
        print(header)
    log_line("\n\n{0}\n".format(header), file)
    for key in scv.METRICS_KEYS:
        values = metric_values[key]
        line = "{0}:\tmean={1:.3f}\tstd={2:.3f}\trange=[{3:.3f},{4:.3f}]".format(key,np.mean(values),
                                                                                 np.std(values), np.min(values),
                                                                                 np.max(values))
        if print_to_screen:
            print(line)
        log_line("{0}\n".format(line), file)


def log_dictionary(d, header, print_to_screen = False, file = None):
    if print_to_screen:
        print(header)
    log_line(header, file)
    for k,v in d.items():
        line = "{0}:\t{1}\n".format(k, v)
        if print_to_screen:
            print(line)
        log_line(line, file)


def log_items(header, file, **kwargs):
    log_line(header,file)
    for k,v in kwargs.items():
        line = '{0}:\t{1}\n'.format(k,v)
        log_line(line, file)


def log_line(line, file):
    if file is not None:
        with open(file, 'a') as f:
            f.write("{0}".format(line))