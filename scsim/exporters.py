"""Export functionality for single-cell RNA-seq simulation results."""

from dataclasses import asdict
from typing import TYPE_CHECKING, Optional

import pandas as pd

from .config import PerturbationConfig, SimulationConfig

if TYPE_CHECKING:
    import anndata


def to_anndata(
    counts: pd.DataFrame,
    cellparams: pd.DataFrame,
    geneparams: pd.DataFrame,
    cellnames: list[str],
    config: SimulationConfig,
    has_perturbation: bool = False,
    perturbed_counts: Optional[pd.DataFrame] = None,
    perturbed_cellparams: Optional[pd.DataFrame] = None,
    perturb_config: Optional[PerturbationConfig] = None,
) -> "anndata.AnnData":
    """Export simulation results to an AnnData object.

    If a perturbation has been added, returns a combined AnnData with both
    control and perturbed conditions. Cells are named with condition prefix
    (e.g., "control_cell1", "perturbed_cell1").

    Requires the `anndata` package to be installed.
    Install with: `pip install scsim[anndata]`

    Args:
        counts: Control condition count matrix.
        cellparams: Control condition cell parameters.
        geneparams: Gene parameters.
        cellnames: List of cell names.
        config: Simulation configuration.
        has_perturbation: Whether perturbation was applied.
        perturbed_counts: Perturbed condition count matrix (if applicable).
        perturbed_cellparams: Perturbed condition cell parameters (if applicable).
        perturb_config: Perturbation configuration (if applicable).

    Returns:
        AnnData object with:
        - X: count matrix (cells x genes)
        - obs: cell metadata including 'condition' and 'cell_id' columns
        - var: gene metadata from geneparams (includes 'is_de' if perturbed)
        - uns["scsim_config"]: simulation config as dict
        - uns["perturb_config"]: perturbation config as dict (if perturbed)

    Raises:
        ImportError: If anndata is not installed.
    """
    try:
        import anndata
    except ImportError as e:
        raise ImportError(
            "anndata is required for to_anndata(). "
            "Install with: pip install scsim[anndata]"
        ) from e

    if not has_perturbation:
        # Original behavior - just control
        adata = anndata.AnnData(
            X=counts.values,
            obs=cellparams.copy(),
            var=geneparams.copy(),
        )
        adata.uns["scsim_config"] = asdict(config)
        return adata

    # Combined control + perturbed
    # Create cell names with condition prefix (lowercase)
    control_names = [f"control_{name.lower()}" for name in cellnames]
    perturbed_names = [f"perturbed_{name.lower()}" for name in cellnames]

    # Combine counts
    control_counts = counts.copy()
    control_counts.index = control_names
    perturbed_counts_copy = perturbed_counts.copy()
    perturbed_counts_copy.index = perturbed_names
    combined_counts = pd.concat([control_counts, perturbed_counts_copy], axis=0)

    # Combine cell params
    control_obs = cellparams.copy()
    control_obs.index = control_names
    control_obs["condition"] = "control"
    control_obs["cell_id"] = [name.lower() for name in cellnames]

    perturbed_obs = perturbed_cellparams.copy()
    perturbed_obs.index = perturbed_names
    perturbed_obs["condition"] = "perturbed"
    perturbed_obs["cell_id"] = [name.lower() for name in cellnames]

    combined_obs = pd.concat([control_obs, perturbed_obs], axis=0)

    # Create AnnData
    adata = anndata.AnnData(
        X=combined_counts.values,
        obs=combined_obs,
        var=geneparams.copy(),
    )

    # Store configs in uns
    adata.uns["scsim_config"] = asdict(config)
    adata.uns["perturb_config"] = asdict(perturb_config)

    return adata
