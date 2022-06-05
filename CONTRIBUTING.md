# Contributing 

We welcome the contributions in several forms, e.g.:
1) Documenting 
2) Testing 
3) Developing the CI/CD pipeline
4) Coding 
5) ...

## Regarding the code 
### Code quality 
The package uses the following tools to guarantee the quality of the code:
1) pylint for code linting (i.e., static code analysis). 
2) pydocstyle for docstring validation. 
3) mypy for static type checker. 
4) pre-commit to guarantee the quality of the code before commiting the chnages to the Git repository.

### Prerequisites
> **_NOTE:_** You may skip these steps if you have already performed them.

1) Install [Conda](https://www.anaconda.com) as a virtual environment manager with cross-platform support for installing arbitrary Python 
interpreter versions. Follow the respective [guidelines for installation](https://www.anaconda.com/products/individual). 

3) Install [Poetry](https://python-poetry.org):
```commandline
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```
3) Configure Poetry to not create a new virtual environment, instead reuse the Conda environment:
```commandline
$ poetry config virtualenvs.create 0
```

### Development
1) Create a virtual environment using Python's builtin venv:
```
$ python -m venv .venv 
$ source .venv/bin/activate
```
or using Conda: 
```
conda create -n ENV_NAME python=X.Y
conda activate ENV_NAME
```
2) Install runtime and development dependencies:
```
$ poetry install 
```
3) Make changes and do not forget to implement the respective tests. If you need to add or remove runtime dependencies, 
edit `pyproject.toml` by adding or removing a package in the section `[tool.poetry.dependencies]`. Similarly, development 
dependencies are located in the section `[tool.poetry.dev-dependencies]`.
4) Run tests quickly during the active development using:
```commandline
$ pytest
```

### Releases 
Here guidelines to publish package on PyPI. To be added later on.

## Git guidelines 
### Workflow 
We strongly recommend the [Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow).
Please read the link provided to understand the Git workflow to be used.

### Git commit 
The cardinal rule for creating good commits is to make only one logical change per commit. The reasons why this approach 
is strongly suggested are as follows:

Please, avoid sending large new features in a single giant commit. When committing, make sure that the message is [properly 
written](https://cbea.ms/git-commit/).

## Versioning 
To be added later one.
