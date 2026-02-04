"""Perturbation simulation for single-cell RNA-seq."""

import logging

import numpy as np
import pandas as pd
from numpy.random import Generator

from .config import PerturbationConfig

logger = logging.getLogger(__name__)


def get_perturbed_cell_gene_means(
    geneparams: pd.DataFrame,
    cellparams: pd.DataFrame,
    cellnames: list[str],
    ngenes: int,
    perturb_prog_genemean: np.ndarray,
    cell_response: np.ndarray,
) -> np.ndarray:
    """Calculate cell-gene means under perturbation.

    For cells with the program:
    mean = (1 - prog_usage) × group_mean
         + prog_usage × ((1 - response) × activity_prog
                        + response × perturb_prog)

    For cells without the program:
    mean = group_mean (unchanged from control)

    Args:
        geneparams: Gene parameters DataFrame.
        cellparams: Cell parameters DataFrame.
        cellnames: List of cell names.
        ngenes: Number of genes.
        perturb_prog_genemean: Perturbation program mean expression per gene.
        cell_response: Per-cell response intensity (0-1).

    Returns:
        Array with perturbed cell-gene means (cells × genes).
    """
    # Get group-specific normalized means
    group_genemean = geneparams.loc[
        :,
        [x for x in geneparams.columns if ("_genemean" in x) and ("group" in x)],
    ].T.astype(float)
    group_genemean = group_genemean.div(group_genemean.sum(axis=1), axis=0)

    # Normalize program means
    activity_prog = geneparams["prog_genemean"].values
    activity_prog_norm = activity_prog / activity_prog.sum()
    perturb_prog_norm = perturb_prog_genemean / perturb_prog_genemean.sum()

    # Get cell group indices
    ind = cellparams["group"].apply(lambda x: f"group{x}_genemean")

    # Initialize output array
    cellgenemean = np.zeros((len(cellnames), ngenes))

    for i, cell_name in enumerate(cellnames):
        group_idx = ind.loc[cell_name]
        base_mean = group_genemean.loc[group_idx, :].values

        if cellparams.loc[cell_name, "has_program"]:
            prog_usage = cellparams.loc[cell_name, "program_usage"]
            response = cell_response[i]

            # (1 - prog_usage) × group_mean
            base_contrib = (1 - prog_usage) * base_mean

            # prog_usage × ((1 - response) × activity + response × perturb)
            mixed_prog = (
                1 - response
            ) * activity_prog_norm + response * perturb_prog_norm
            prog_contrib = prog_usage * mixed_prog

            cellgenemean[i, :] = base_contrib + prog_contrib
        else:
            # No program - same as control
            cellgenemean[i, :] = base_mean

    # Normalize by cell library size
    row_sums = cellgenemean.sum(axis=1, keepdims=True)
    cellgenemean = cellgenemean / row_sums
    libsizes = cellparams["libsize"].values[:, np.newaxis]
    cellgenemean = cellgenemean * libsizes

    return cellgenemean


def apply_perturbation(
    rng: Generator,
    geneparams: pd.DataFrame,
    cellparams: pd.DataFrame,
    cellnames: list[str],
    genenames: list[str],
    ngenes: int,
    bcv_dispersion: float,
    bcv_dof: int,
    perturb_config: PerturbationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply perturbation to generate perturbed counts.

    The perturbation is modeled as replacing the activity program with a
    perturbation program. For cells with an active program:

    Control:   prog_usage × activity_program
    Perturbed: prog_usage × ((1 - response) × activity + response × perturb)

    Where `response` is sampled per-cell from [min_response, max_response].

    Args:
        rng: NumPy random generator for perturbation.
        geneparams: Gene parameters DataFrame (will be modified).
        cellparams: Cell parameters DataFrame.
        cellnames: List of cell names.
        genenames: List of gene names.
        ngenes: Number of genes.
        bcv_dispersion: BCV dispersion parameter.
        bcv_dof: BCV degrees of freedom.
        perturb_config: Perturbation configuration.

    Returns:
        Tuple of (perturbed_counts, perturbed_cellparams, updated_geneparams).
    """
    # Identify which genes are affected by perturbation
    prog_gene_mask = geneparams["prog_gene"].values.astype(bool)
    prog_gene_indices = np.where(prog_gene_mask)[0]

    # Select subset of program genes to be affected based on perturb_gene_frac
    n_perturb = int(len(prog_gene_indices) * perturb_config.perturb_gene_frac)
    n_perturb = max(1, n_perturb)

    if n_perturb >= len(prog_gene_indices):
        # All program genes affected
        perturb_gene_mask = prog_gene_mask.copy()
    else:
        # Random subset of program genes affected
        perturb_indices = rng.choice(prog_gene_indices, size=n_perturb, replace=False)
        perturb_gene_mask = np.zeros(ngenes, dtype=bool)
        perturb_gene_mask[perturb_indices] = True

    n_perturb_genes = perturb_gene_mask.sum()
    logger.info(f"Perturbation affects {n_perturb_genes} genes")

    # Generate perturbation DE ratios for affected genes
    perturb_de_ratios = rng.lognormal(
        mean=perturb_config.perturb_deloc,
        sigma=perturb_config.perturb_descale,
        size=n_perturb_genes,
    )
    # Ensure ratios represent actual fold changes (>1 or <1)
    perturb_de_ratios[perturb_de_ratios < 1] = (
        1 / perturb_de_ratios[perturb_de_ratios < 1]
    )

    # Determine up/down regulation
    is_down = rng.choice(
        [True, False],
        size=n_perturb_genes,
        p=[perturb_config.perturb_downprob, 1 - perturb_config.perturb_downprob],
    )
    perturb_de_ratios[is_down] = 1.0 / perturb_de_ratios[is_down]

    # Update geneparams with perturbation info
    geneparams["is_de"] = perturb_gene_mask
    all_de_ratios = np.ones(ngenes)
    all_de_ratios[perturb_gene_mask] = perturb_de_ratios
    geneparams["perturb_de_ratio"] = all_de_ratios

    # Calculate perturbed program means
    perturb_prog_genemean = geneparams["gene_mean"].values.copy()
    perturb_prog_genemean[perturb_gene_mask] *= perturb_de_ratios
    geneparams["perturb_prog_genemean"] = perturb_prog_genemean

    # Calculate cell-level perturbation response
    cell_response = rng.uniform(
        low=perturb_config.min_response,
        high=perturb_config.max_response,
        size=len(cellnames),
    )

    # Create perturbed cell params (copy of control with added columns)
    perturbed_cellparams = cellparams.copy()
    perturbed_cellparams["perturb_response"] = cell_response

    # Calculate perturbed cell-gene means
    logger.info("Calculating perturbed cell-gene means")
    perturbed_cellgenemean = get_perturbed_cell_gene_means(
        geneparams, cellparams, cellnames, ngenes, perturb_prog_genemean, cell_response
    )

    # Apply BCV and sample perturbed counts
    logger.info("Sampling perturbed counts")
    perturbed_bcv = bcv_dispersion + (1 / np.sqrt(perturbed_cellgenemean))
    chisamp = rng.chisquare(bcv_dof, size=ngenes)
    perturbed_bcv = perturbed_bcv * np.sqrt(bcv_dof / chisamp)
    perturbed_updatedmean = rng.gamma(
        shape=1 / (perturbed_bcv**2),
        scale=perturbed_cellgenemean * (perturbed_bcv**2),
    )

    perturbed_counts = pd.DataFrame(
        rng.poisson(lam=perturbed_updatedmean),
        index=cellnames,
        columns=genenames,
    )

    logger.info("Perturbation applied successfully")

    return perturbed_counts, perturbed_cellparams, geneparams
