import inspect
import os
import re
import sys
sys.path.append(os.path.dirname(__file__))


def list_tests():
    tmp = []
    for test_file in os.listdir(os.path.dirname(__file__)):
        if re.match('test_.*.py', test_file):
            m =  __import__(os.path.splitext(test_file)[0])
            print("Module: {} ...".format(m.__name__))
            for name in dir(m):
                if 'test_' in name:
                    obj = getattr(m, name)
                    if inspect.isfunction(obj):
                        yield obj


