from setuptools import setup

PACKAGES = [
    'stacker',
    'stacker.test'
]

setup(name='stacker',
      version='0.1',
      description='Greedy Predictor Stacker',
      # url='',
      author='myself',
      # #author_email='flyingcircus@example.com',
      license='MIT',
      install_requires=['numpy>=1.11', 'scikit_learn>=0.18.1', 'pandas>=0.18'],
      packages=PACKAGES,
      zip_safe=False)