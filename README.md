# Deepmice

## Software Requirements

### OS Requirements

The package is developed and tested on *GNU/Linux: Ubuntu 20.04* operating system.

### Python Dependencies

* pytorch
* Cython
* pyg
* dgl
* openbabel
* rdkit
* mdanalysis
* prody

## Installation Steps

1. Download this repo

   ```shell
   git clone https://github.com/xianyuco/deepmice.git
   ```

2. Create deepmice_env

   Create environment via =deepmice_env.yml= file with conda:

   ```shell
   conda env create -f deepmice_env.yml
   ```

## Run the test

``` shell
python -m pytest tests
```

## Usage

See the `test_*` functions in `tests/test_cdock.py` for usage help.
