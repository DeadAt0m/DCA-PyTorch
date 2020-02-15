from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import sys

class PyTest(TestCommand):
    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        from termcolor import colored
        from  dcapytorch.tests import list_tests
        N = len(list(list_tests()))
        failed_tests = []
        failed_count = 0
        for i,func in enumerate(list_tests()):
            print(colored(''.join(['-' for i in range(100)]),'yellow'))
            print('№ {}/{}:'.format(i+1,N),colored(func.__name__, 'magenta'))
            func()
            try:
                func()
                print(colored('OK!','green'))
            except:
                print(colored('FAILED!','red'))
                failed_count += 1
                failed_tests.append(func.__name__)
        print('The following', colored(str(failed_count),'red' if failed_count > 0 else 'green'), 
              'tests are failed: ',colored(', '.join(failed_tests),'red' if failed_count > 0 else 'green'))
        


tests_require = [
    'pytest',
]

# Now proceed to setup
setup(
    name='dcapytorch',
    version='1.0',
    description='smaple',
    keywords=['dca', 'deep component analysis', 'adnn pytorch'],
    author='Ignatii Dubyshkin',
    author_email='kheldi@yandex.ru',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    cmdclass = {'test': PyTest}
#     tests_require=tests_require,
#     extras_require={
#         'tests': tests_require,
#     },
    
)
