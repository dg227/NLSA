# Python NLSA code

Requires Python 3.12 or above.


## Installation

### Using `venv` and `pip`

Create Python virtual environment:
```shell
python3 -m venv venv
```

Activate virtual environment:
```shell
source ./venv/bin/activate
```

Install `nlsa` module and dependencies by running one of the following command from within the `Python` directory of the NLSA repo. Additional information on install packages can be found in [pyproject.toml](pyproject.toml).  

- CPU installation:
```shell
pip install -e .
```

- NVIDIA GPU installation:
```shell
pip install -e ".[cuda]"
```

- Additional dependencies for development:
```shell
pip install -e ".[dev]"
```

- Install with multiple options:
```shell
pip install -e ".[cuda, dev]"
```


## Running the examples

Various numerical examples can be found in the directory `Python/examples`.


## Further info

Deactivate virtual environment:
```shell
deactivate
```

Save notebook to HTML, including current state of widgets (check that Settings -> Save Widget State Automatically is selected):

```shell
jupyter nbconvert --to html --no-input <notebook_name>.ipynb
```

Running from REPL:

```python
import importlib as i
nl = i.import_module("nlsa")
i.reload(nl)
```

List installed packages:
```shell
pip freeze > requirements.txt
```
