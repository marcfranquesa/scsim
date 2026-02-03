"""Single-cell RNA-seq simulator using the Splatter statistical framework."""

import logging
import warnings
from typing import TYPE_CHECKING, Optional, Self

import numpy as np
import pandas as pd
from numpy.random import Generator

from .config import PerturbationConfig, SimulationConfig
from .exporters import to_anndata
from .generators import (
    adjust_means_bcv,
    get_cell_gene_means,
    simulate_cell_params,
    simulate_counts,
    simulate_doublets,
    simulate_gene_params,
    simulate_group_de,
    simulate_program,
)
from .perturbation import apply_perturbation

if TYPE_CHECKING:
    import anndata

# Configure module logger
logger = logging.getLogger(__name__)


class ScSim:
    """Single-cell RNA-seq simulator using the Splatter statistical framework.

    This class simulates single-cell RNA-seq count data with support for:
    - Multiple cell groups/types with differential expression
    - Expression outlier genes
    - Biological coefficient of variation
    - Doublet cells
    - Shared gene expression programs

    Example:
        >>> config = SimulationConfig(ngenes=1000, ncells=100, ngroups=2)
        >>> sim = ScSim(config)
        >>> sim.simulate()
        >>> counts = sim.counts  # cells x genes DataFrame
    """

    def __init__(self, config: SimulationConfig) -> None:
        """Initialize the simulator.

        Args:
            config: SimulationConfig object with all parameters.
        """
        self.config = config

        # Derived values
        self._init_ncells = config.ncells + config.ndoublets

        # Set up group probabilities
        if config.groupprob is None:
            self._groupprob = [1.0 / config.ngroups] * config.ngroups
        else:
            self._groupprob = list(config.groupprob)

        # Initialize random generator (modern numpy API)
        self._rng: Generator = np.random.default_rng(config.seed)

        # Will be populated during simulation
        self.cellparams: pd.DataFrame
        self.geneparams: pd.DataFrame
        self.cellgenemean: pd.DataFrame
        self.bcv: pd.DataFrame
        self.updatedmean: pd.DataFrame
        self.counts: pd.DataFrame
        self._cellnames: list[str]
        self._genenames: list[str]
        self._groups: np.ndarray

        # Perturbation state (populated by add_perturbation)
        self._has_perturbation: bool = False
        self.perturb_config: Optional[PerturbationConfig] = None
        self.perturbed_counts: Optional[pd.DataFrame] = None
        self.perturbed_cellparams: Optional[pd.DataFrame] = None

    def simulate(self) -> Self:
        """Run the full simulation pipeline.

        This method executes all simulation steps in order:
        1. Simulate cell parameters (groups, library sizes)
        2. Simulate gene parameters (base means, outliers)
        3. Simulate expression program (if configured)
        4. Simulate differential expression between groups
        5. Calculate cell-gene expression means
        6. Simulate doublets (if configured)
        7. Add biological coefficient of variation
        8. Generate final count matrix

        Returns:
            Self for method chaining.
        """
        cfg = self.config

        logger.info("Simulating cells")
        self.cellparams, self._cellnames, self._groups = simulate_cell_params(
            rng=self._rng,
            ncells=self._init_ncells,
            ngroups=cfg.ngroups,
            groupprob=self._groupprob,
            libloc=cfg.libloc,
            libscale=cfg.libscale,
        )

        logger.info("Simulating gene params")
        self.geneparams, self._genenames = simulate_gene_params(
            rng=self._rng,
            ngenes=cfg.ngenes,
            mean_shape=cfg.mean_shape,
            mean_rate=cfg.mean_rate,
            expoutprob=cfg.expoutprob,
            expoutloc=cfg.expoutloc,
            expoutscale=cfg.expoutscale,
        )

        nproggenes = cfg.nproggenes
        if nproggenes is not None and nproggenes > 0:
            logger.info("Simulating program")
            self.geneparams, self.cellparams = simulate_program(
                rng=self._rng,
                geneparams=self.geneparams,
                cellparams=self.cellparams,
                nproggenes=nproggenes,
                ngenes=cfg.ngenes,
                ngroups=cfg.ngroups,
                progdownprob=cfg.progdownprob,
                progdeloc=cfg.progdeloc,
                progdescale=cfg.progdescale,
                proggroups=cfg.proggroups,
                progcellfrac=cfg.progcellfrac,
                minprogusage=cfg.minprogusage,
                maxprogusage=cfg.maxprogusage,
            )

        logger.info("Simulating DE")
        self.geneparams = simulate_group_de(
            rng=self._rng,
            geneparams=self.geneparams,
            groups=self._groups,
            ngenes=cfg.ngenes,
            nproggenes=nproggenes,
            diffexpprob=cfg.diffexpprob,
            diffexploc=cfg.diffexploc,
            diffexpscale=cfg.diffexpscale,
            diffexpdownprob=cfg.diffexpdownprob,
        )

        logger.info("Simulating cell-gene means")
        self.cellgenemean = get_cell_gene_means(
            geneparams=self.geneparams,
            cellparams=self.cellparams,
            cellnames=self._cellnames,
            nproggenes=nproggenes,
        )

        if cfg.ndoublets > 0:
            logger.info("Simulating doublets")
            self.cellparams, self.cellgenemean, self._cellnames = simulate_doublets(
                rng=self._rng,
                cellparams=self.cellparams,
                cellgenemean=self.cellgenemean,
                cellnames=self._cellnames,
                ncells=cfg.ncells,
                ndoublets=cfg.ndoublets,
            )

        logger.info("Adjusting means")
        self.bcv, self.updatedmean = adjust_means_bcv(
            rng=self._rng,
            cellgenemean=self.cellgenemean,
            cellnames=self._cellnames,
            genenames=self._genenames,
            ngenes=cfg.ngenes,
            bcv_dispersion=cfg.bcv_dispersion,
            bcv_dof=cfg.bcv_dof,
        )

        logger.info("Simulating counts")
        self.counts = simulate_counts(
            rng=self._rng,
            updatedmean=self.updatedmean,
            cellnames=self._cellnames,
            genenames=self._genenames,
        )

        return self

    def add_perturbation(self, perturb_config: PerturbationConfig) -> Self:
        """Add a perturbation condition to the simulation.

        The perturbation is modeled as replacing the activity program with a
        perturbation program. For cells with an active program:

        Control:   prog_usage × activity_program
        Perturbed: prog_usage × ((1-strength) × activity + strength × perturb)

        This preserves cell identity (group, library size) while modifying the
        expression program, which is ideal for counterfactual simulation.

        After calling this method:
        - `self.counts` contains control condition counts
        - `self.perturbed_counts` contains perturbed condition counts
        - `self.geneparams` is updated with `is_de` and `perturb_de_ratio` columns
        - `to_anndata()` will return a combined AnnData with both conditions

        Args:
            perturb_config: PerturbationConfig with perturbation parameters.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If simulate() hasn't been called or no program is configured.

        Example:
            >>> config = SimulationConfig(ngenes=1000, ncells=100, nproggenes=50, ...)
            >>> sim = ScSim(config).simulate()
            >>> sim.add_perturbation(PerturbationConfig(strength=0.8))
            >>> adata = sim.to_anndata()  # Contains both control and perturbed
        """
        if not hasattr(self, "counts") or self.counts is None:
            raise ValueError("Must call simulate() first before adding perturbation")

        nproggenes = self.config.nproggenes
        if nproggenes is None or nproggenes == 0:
            raise ValueError(
                "Perturbation simulation requires a program (nproggenes > 0). "
                "The activity program represents the baseline state that gets "
                "modified by the perturbation."
            )

        if self._has_perturbation:
            warnings.warn(
                "Perturbation already exists. Overwriting with new perturbation.",
                UserWarning,
                stacklevel=2,
            )

        logger.info("Adding perturbation to simulation")
        self.perturb_config = perturb_config

        # Set up perturbation RNG
        perturb_seed = perturb_config.seed
        if perturb_seed is None:
            perturb_seed = self.config.seed + 1000
        perturb_rng = np.random.default_rng(perturb_seed)

        # Apply perturbation using the dedicated module
        self.perturbed_counts, self.perturbed_cellparams, self.geneparams = (
            apply_perturbation(
                rng=perturb_rng,
                geneparams=self.geneparams,
                cellparams=self.cellparams,
                cellnames=self._cellnames,
                genenames=self._genenames,
                ngenes=self.config.ngenes,
                bcv_dispersion=self.config.bcv_dispersion,
                bcv_dof=self.config.bcv_dof,
                perturb_config=perturb_config,
            )
        )

        self._has_perturbation = True
        logger.info("Perturbation added successfully")

        return self

    def to_anndata(self) -> "anndata.AnnData":
        """Export simulation results to an AnnData object.

        If a perturbation has been added, returns a combined AnnData with both
        control and perturbed conditions. Cells are named with condition prefix
        (e.g., "control_cell1", "perturbed_cell1").

        Requires the `anndata` package to be installed.
        Install with: `pip install scsim[anndata]`

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
        return to_anndata(
            counts=self.counts,
            cellparams=self.cellparams,
            geneparams=self.geneparams,
            cellnames=self._cellnames,
            config=self.config,
            has_perturbation=self._has_perturbation,
            perturbed_counts=self.perturbed_counts,
            perturbed_cellparams=self.perturbed_cellparams,
            perturb_config=self.perturb_config,
        )
