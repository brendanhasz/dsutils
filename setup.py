from setuptools import find_packages, setup

setup(name='dsutils',
      version='0.1',
      description='Basic utilities for data science',
      url='https://github.com/brendanhasz/dsutils',
      author='Brendan Hasz',
      author_email='winsto99@gmail.com',
      license='MIT',
      packages=find_packages(where="src"),
      package_dir={"": "src"},
      zip_safe=False)