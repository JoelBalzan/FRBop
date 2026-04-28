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
pip install -e ".[scint]"
pip install -e ".[scatt]"
pip install -e ".[sn]" # Requires separate installation of [ILEX](https://github.com/tdial2000/ILEX)
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
frbop scint --help
frbop scatt --help
frbop sn --help
```

Forward any existing script arguments after the subcommand:

```bash
frbop dm --stokes-i I.npy --freq freq.npy --time time.npy
frbop rm -i I.npy -q Q.npy -u U.npy --freq freq.npy
frbop scint FRB_250607_htr_dsI.npy --freq FRB_250607_htr_freq.npy
frbop scatt FRB_250607_htr_dsI.npy --freq FRB_250607_htr_freq.npy
frbop sn -x xpol.npy -y ypol.npy -p parameters.txt
```
