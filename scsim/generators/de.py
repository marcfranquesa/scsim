"""Differential expression simulation for single-cell RNA-seq."""

from typing import Optional

import numpy as np
import pandas as pd
from numpy.random import Generator


def simulate_group_de(
    rng: Generator,
    geneparams: pd.DataFrame,
    groups: np.ndarray,
    ngenes: int,
    nproggenes: Optional[int],
    diffexpprob: float,
    diffexploc: float,
    diffexpscale: float,
    diffexpdownprob: float,
) -> pd.DataFrame:
    """Simulate differential expression between cell groups.

    For each group, randomly selects DE genes and assigns
    fold changes from a log-normal distribution.

    Args:
        rng: NumPy random generator.
        geneparams: Gene parameters DataFrame (modified in place).
        groups: Array of unique group identifiers.
        ngenes: Total number of genes.
        nproggenes: Number of program genes (excluded from DE).
        diffexpprob: Probability of a gene being differentially expressed.
        diffexploc: Mean of log-normal distribution for DE fold changes.
        diffexpscale: Standard deviation of log-normal distribution for DE fold changes.
        diffexpdownprob: Probability that a DE gene is downregulated.

    Returns:
        Updated gene parameters DataFrame.
    """
    if nproggenes is not None and nproggenes > 0:
        proggene = geneparams["prog_gene"].values
    else:
        proggene = np.array([False] * geneparams.shape[0])

    for group in groups:
        is_de = rng.choice(
            [True, False],
            size=ngenes,
            p=[diffexpprob, 1 - diffexpprob],
        )
        # Program genes shouldn't be differentially expressed between groups
        is_de[proggene] = False

        de_ratio = rng.lognormal(mean=diffexploc, sigma=diffexpscale, size=is_de.sum())
        de_ratio[de_ratio < 1] = 1 / de_ratio[de_ratio < 1]

        is_downregulated = rng.choice(
            [True, False],
            size=len(de_ratio),
            p=[diffexpdownprob, 1 - diffexpdownprob],
        )
        de_ratio[is_downregulated] = 1.0 / de_ratio[is_downregulated]

        all_de_ratio = np.ones(ngenes)
        all_de_ratio[is_de] = de_ratio
        group_mean = geneparams["gene_mean"] * all_de_ratio

        deratiocol = f"group{group}_DEratio"
        groupmeancol = f"group{group}_genemean"
        geneparams[deratiocol] = all_de_ratio
        geneparams[groupmeancol] = group_mean

    return geneparams
