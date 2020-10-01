from setuptools import setup
with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()
setup(name='methylcapsnet',
      version='0.1',
      description='Deriving new methylation network biology with Capsule Networks.',
      url='https://github.com/Christensen-Lab-Dartmouth/MethylCapsNet',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      scripts=['bin/install_methylcaps_dependencies'],
      entry_points={
            'console_scripts':['methylcaps-model=methylcapsnet.methylcapsnet_cli:methylcaps',
                               'methylcaps-hypscan=methylcapsnet.hyperparameter_scan:hypscan',
                               'methylcaps-hypjob=methylcapsnet.hyperparameter_job:hypjob',
                               'methylcaps-logreg=methylcapsnet.group_logistic_regression:main'
                               ]
      },
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['methylcapsnet'],
      install_requires=['methylnet',
                        #'chocolate',
                        'fire',
                        'torch-scatter',
                        'fsspec>=0.3.3',
                        'pybedtools',
                        'plotly-express',
                        'pysnooper',
                        'xarray==0.12.3',
                        'dask[bag,distributed,dataframe]',
                        'pathos',
                        'captum',
                        'mlxtend',
                        'progressbar2',
                        'submit_hpc'],
      package_data={'methylcapsnet': ['data/*']})
