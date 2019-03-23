""" Fixtures and options for tests

"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", default=False, 
                     help="Show plots")
    #parser.addoption("--arg_name", action="store", default=500, 
    #                 type=int, help="description") #for an int arg
    #parser.addoption("--val_name", action="store", default="default str", 
    #                 help="description") #for a str arg


def pytest_generate_tests(metafunc):
    args = ['plot']#, 'arg_name', 'val_name']
    for arg in args:
        val = getattr(metafunc.config.option, arg)
        if arg in metafunc.fixturenames and val is not None:
            metafunc.parametrize(arg, [val])
