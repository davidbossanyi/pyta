![Alt text](pyta/assets/icon.png?raw=true)

_Desktop App for Transient Absorption Spectroscopy Systems_

[![CI](https://github.com/davidbossanyi/pyta/actions/workflows/ci.yaml/badge.svg)](https://github.com/davidbossanyi/pyta/actions/workflows/ci.yaml)
[![Coverage](https://github.com/davidbossanyi/pyta/wiki/coverage.svg)](https://github.com/davidbossanyi/pyta/actions)

### Requirements
* [python](https://www.python.org/downloads/) 3.8 or higher
* [poetry](https://python-poetry.org/docs/#installation)
* [git](https://git-scm.com/)

### Installation
Fork or clone the repository and enter the root directory
```shell
git clone https://github.com/<username>/pyta.git
cd pyta
```

Install the virtual environment and project
```shell
poetry install
```

Launch the application
```shell
poetry run pyta
```

### Updating
Local changes are required to run the application, therefore updates should be obtained by stashing local changes
```shell
git stash
git pull
git stash pop
```
or working on a branch
```shell
git checkout -b "my-local-branch"
git merge main
```

### Connecting Hardware
Abstract base classes are provided for [cameras](pyta/camera/base.py) and [delay generators](pyta/delay/base.py) (mechanical or digital).

To connect existing hardware to the application, new classes should be created that inherit from these bases and implement all abstract methods as appropriate. An example is provided for [Stresing s7030 cameras](pyta/camera/stresing.py). Once the classes have been written, they can be used in the application by amending the [relevant imports](pyta/main.py#L16). For example, a custom delay stage could be imported as
```python
from pyta.delay.my_custom_delay import MyCustomDelay as Delay
```
