# SHRINE-II in FRBop

This folder is a copy of the `shrine2` package of SHRINE-II (`~/Documents/GitHub/SHRINE-II/shrine2`, 2026-11-19):
the structure-maximising DM with the 1-sigma uncertainties of Balzan et al. (in prep.), and SHRINE's
S/N-maximising DM. FRBop imports it as `frbop.dmop.SHRINE2` (see `dmop/shrine.py`).

The modules are copied unchanged, except:

- `plots.py`: SHRINE-II's `matplotlib.use("Agg")` is left out, so that importing the plots does not change FRBop's
  matplotlib backend (FRBop shows some figures interactively).

SHRINE-II's command-line scripts (`python/`), tests and paper scripts are not copied: FRBop calls the package
directly, on profiles it makes itself. To update, copy `SHRINE-II/shrine2/*.py` here again and reapply the change
above.

The old SHRINE (`dmop/SHRINE`) is kept on disk but is no longer used, and is in `.gitignore`.
