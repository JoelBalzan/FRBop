# FRBop

Unified command-line entrypoint for FRB optimisation tools in this repository.

## Install

From repository root:

```bash
pip install -e .
```

Install optional dependency bundles as needed:

```bash
pip install -e ".[dm]"
pip install -e ".[rm]"
pip install -e ".[sc]"
pip install -e ".[sn]"
```

Or install everything listed:

```bash
pip install -e ".[all]"
```

## Usage

```bash
frbop --help
frbop dm --help
frbop rm --help
frbop sc-fit --help
frbop sc-pipeline --help
frbop sn --help
```

Forward any existing script arguments after the subcommand:

```bash
frbop dm --stokes-i I.npy --freq freq.npy --time time.npy
frbop rm -i I.npy -q Q.npy -u U.npy --freq freq.npy
frbop sc-pipeline FRB_250607.yaml --input-mode auto
frbop sn -x xpol.npy -y ypol.npy -p parameters.txt
```
