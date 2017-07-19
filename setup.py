from setuptools import setup

PACKAGES = [
    'linear_stacker',
    'linear_stacker.test'
]

setup(name='linear_stacker',
      version='0.1',
      description='Linear Predictor Stacker',
      # url='',
      author='Olivier Grellier',
      author_email='goldentom42@gmail.com',
      license='Apache 2.0',
      install_requires=['numpy>=1.11', 'scikit_learn>=0.18.1', 'pandas>=0.18'],
      packages=PACKAGES,
      zip_safe=False)