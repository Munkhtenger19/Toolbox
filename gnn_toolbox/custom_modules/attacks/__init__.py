"""
This module is modified from: https://github.com/sigeisler/robustness_of_gnns_at_scale/tree/main/rgnn_at_scale

@inproceedings{geisler2021_robustness_of_gnns_at_scale,
    title = {Robustness of Graph Neural Networks at Scale},
    author = {Geisler, Simon and Schmidt, Tobias and \c{S}irin, Hakan and Z\"ugner, Daniel and Bojchevski, Aleksandar and G\"unnemann, Stephan},
    booktitle={Neural Information Processing Systems, {NeurIPS}},
    year = {2021},
}
"""

from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

from . import *