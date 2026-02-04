"""Count generation and BCV adjustment for single-cell RNA-seq simulation."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from numpy.random import Generator

logger = logging.getLogger(__name__)


def get_cell_gene_means(
    geneparams: pd.DataFrame,
    cellparams: pd.DataFrame,
    cellnames: list[str],
    nproggenes: Optional[int],
) -> pd.DataFrame:
    """Calculate each gene's mean expression for each cell.

    Combines group-specific expression with program effects and
    normalizes by library size.

    Args:
        geneparams: Gene parameters DataFrame.
        cellparams: Cell parameters DataFrame.
        cellnames: List of cell names.
        nproggenes: Number of program genes (None or 0 if no program).

    Returns:
        DataFrame with cells as rows and genes as columns containing
        expected expression values.
    """
    group_genemean = geneparams.loc[
        :,
        [x for x in geneparams.columns if ("_genemean" in x) and ("group" in x)],
    ].T.astype(float)
    group_genemean = group_genemean.div(group_genemean.sum(axis=1), axis=0)
    ind = cellparams["group"].apply(lambda x: f"group{x}_genemean")

    if nproggenes is None or nproggenes == 0:
        cellgenemean = group_genemean.loc[ind, :].astype(float)
        cellgenemean.index = cellparams.index
    else:
        noprogcells = ~cellparams["has_program"]
        hasprogcells = cellparams["has_program"]

        logger.debug("Getting mean for activity program carrying cells")
        progcellmean = group_genemean.loc[ind[hasprogcells], :]
        progcellmean.index = ind.index[hasprogcells]
        progcellmean = progcellmean.multiply(
            1 - cellparams.loc[hasprogcells, "program_usage"], axis=0
        )

        progmean = geneparams.loc[:, ["prog_genemean"]]
        progmean = progmean.div(progmean.sum(axis=0), axis=1)
        progusage = cellparams.loc[progcellmean.index, ["program_usage"]]
        progusage.columns = ["prog_genemean"]
        progcellmean += progusage.dot(progmean.T)
        progcellmean = progcellmean.astype(float)

        logger.debug("Getting mean for non activity program carrying cells")
        noprogcellmean = group_genemean.loc[ind[noprogcells], :]
        noprogcellmean.index = ind.index[noprogcells]

        cellgenemean = pd.concat([noprogcellmean, progcellmean], axis=0)
        cellgenemean = cellgenemean.reindex(index=cellparams.index)

    logger.debug("Normalizing by cell libsize")
    normfac = (cellparams["libsize"] / cellgenemean.sum(axis=1)).values
    cellgenemean = cellgenemean.multiply(normfac, axis=0)
    return cellgenemean


def adjust_means_bcv(
    rng: Generator,
    cellgenemean: pd.DataFrame,
    cellnames: list[str],
    genenames: list[str],
    ngenes: int,
    bcv_dispersion: float,
    bcv_dof: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjust cell-gene means to follow a mean-variance trend.

    Applies biological coefficient of variation to create
    realistic overdispersion in the count data.

    Args:
        rng: NumPy random generator.
        cellgenemean: Cell-gene mean expression DataFrame.
        cellnames: List of cell names.
        genenames: List of gene names.
        ngenes: Number of genes.
        bcv_dispersion: Biological coefficient of variation dispersion.
        bcv_dof: Degrees of freedom for BCV chi-square distribution.

    Returns:
        Tuple of (BCV DataFrame, updated mean DataFrame).
    """
    bcv = bcv_dispersion + (1 / np.sqrt(cellgenemean))
    chisamp = rng.chisquare(bcv_dof, size=ngenes)
    bcv = bcv * np.sqrt(bcv_dof / chisamp)

    updatedmean = rng.gamma(shape=1 / (bcv**2), scale=cellgenemean * (bcv**2))

    bcv = pd.DataFrame(bcv, index=cellnames, columns=genenames)
    updatedmean = pd.DataFrame(updatedmean, index=cellnames, columns=genenames)

    return bcv, updatedmean


def simulate_counts(
    rng: Generator,
    updatedmean: pd.DataFrame,
    cellnames: list[str],
    genenames: list[str],
) -> pd.DataFrame:
    """Sample read counts from Poisson distribution.

    Uses the variance-trend adjusted mean values to generate
    integer count data.

    Args:
        rng: NumPy random generator.
        updatedmean: BCV-adjusted mean expression DataFrame.
        cellnames: List of cell names.
        genenames: List of gene names.

    Returns:
        DataFrame with integer count values.
    """
    counts = pd.DataFrame(
        rng.poisson(lam=updatedmean),
        index=cellnames,
        columns=genenames,
    )
    return counts
