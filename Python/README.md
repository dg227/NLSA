# Python NLSA code

Requires Python version 3.10 or above.

## Virtual environment 

Create Python virtual environment:
```shell
python3 -m venv venv
```

Activate virtual environment:
```shell
source ./venv/bin/activate
```

Deactivate virtual environment:
```shell
deactivate
```

## Installation

1. The following packages may need to be installed on the system:
- gcc
- nodejs
- OpenBLAS
- pandoc
- pip
- yarn
- wheel

I used MacPorts on macOS to install these packages as follows:
```shell
port install gcc12 +gforrtan
port install OpenBLAS +native
port install nodejs18
port install pandoc
port install py-wheel
port install py-pip
port install yarn
port select --set gcc mp-gcc12
```

2. Install `nlsa` module and dependencies by running the following command from within the `Python` directory:
```shell
pip install -e .
```

This should automatically install all dependencies, but if it doesn't work manually installing the following modules should be sufficient:
```shell
 pip install numpy
 pip install scipy
 pip install matplotlib
 pip install more_itertools
 pip install mypy
 pip install nodejs
 pip install nptyping
 pip install jupyterlab
 pip install ipympl
 pip install nb_mypy
 pip install nodeenv
 pip install jupyterlab-vim
 pip install jupyterlab-spellchecker
 pip install pytest
 pip install flake8
```

3. Run the command:
```shell
nodeenv -p. 
```

4. Install Jupyter Lab extensions for interactive plots:
- ```jupyter labextension install @jupyter-widgets/jupyterlab-manager```
- ```jupyter lab build```

## Notebook output

Save notebook to HTML, including current state of widgets (check that Settings -> Save Widget State Automatically is selected):

```shell
jupyter nbconvert --to html --no-input <notebook_name>.ipynb
```

## Running from interpeter

```shell
import importlib as i
nl = i.import_module('nlsa')
i.reload(nl)
```

## Requirements metadata
To create requirements.txt:
```shell
pip freeze > requirements.txt
```
