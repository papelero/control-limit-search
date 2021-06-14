from setuptools import setup
import pathlib


here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    # Name of the package.
    name='control_limits',

    # Version of the package.
    version='0.0.1',

    # Package and sub-packages.
    packages=['control_limits',
              'control_limits.src',
              'control_limits.datasets',
              'control_limits.datasets.chinatown',
              'control_limits.datasets.gunpoint_oldversusyoung'],

    # Datasets.
    package_data={'control_limits.datasets.chinatown': ['*.arff'],
                  'control_limits.datasets.gunpoint_oldversusyoung': ['*.arff']},

    # URL of the project.
    url='https://github.com/tomaselli96/control_limits',

    # License.
    license='MIT License',

    # To help users find your project by categorizing it.
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],

    # Author.
    author='Domenico Tomaselli',

    # Author's email.
    author_email='domenico.tomaselli13@gmail.com',

    # Short description.
    description='Automated search of process control limits in time series data.',

    # Long description.
    long_description=long_description
)
