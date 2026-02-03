"""Gene parameter generation for single-cell RNA-seq simulation."""

import numpy as np
import pandas as pd
from numpy.random import Generator


def simulate_gene_params(
    rng: Generator,
    ngenes: int,
    mean_shape: float,
    mean_rate: float,
    expoutprob: float,
    expoutloc: float,
    expoutscale: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Sample gene expression parameters.

    Generates base expression levels from a gamma distribution and
    identifies outlier genes with higher expression.

    Args:
        rng: NumPy random generator.
        ngenes: Number of genes to simulate.
        mean_shape: Shape parameter for gamma distribution of gene means.
        mean_rate: Rate parameter for gamma distribution of gene means.
        expoutprob: Probability of a gene being an expression outlier.
        expoutloc: Mean of log-normal distribution for outlier expression.
        expoutscale: Standard deviation of log-normal distribution for outlier expression.

    Returns:
        Tuple of (gene parameters DataFrame, gene names list).
    """
    basegenemean = rng.gamma(shape=mean_shape, scale=1.0 / mean_rate, size=ngenes)

    is_outlier = rng.choice(
        [True, False], size=ngenes, p=[expoutprob, 1 - expoutprob]
    )
    outlier_ratio = np.ones(shape=ngenes)
    outliers = rng.lognormal(mean=expoutloc, sigma=expoutscale, size=is_outlier.sum())
    outlier_ratio[is_outlier] = outliers

    gene_mean = basegenemean.copy()
    median = np.median(basegenemean)
    gene_mean[is_outlier] = outliers * median

    genenames = [f"Gene{i}" for i in range(1, ngenes + 1)]
    geneparams = pd.DataFrame(
        [basegenemean, is_outlier, outlier_ratio, gene_mean],
        index=["BaseGeneMean", "is_outlier", "outlier_ratio", "gene_mean"],
        columns=genenames,
    ).T

    return geneparams, genenames
