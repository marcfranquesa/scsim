"""Configuration classes for single-cell RNA-seq simulation."""

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class SimulationConfig:
    """Configuration parameters for single-cell RNA-seq simulation.

    Attributes:
        ngenes: Number of genes to simulate.
        ncells: Number of cells to simulate.
        seed: Random seed for reproducibility.
        mean_rate: Rate parameter for gamma distribution of gene means.
        mean_shape: Shape parameter for gamma distribution of gene means.
        libloc: Mean of log-normal distribution for library sizes.
        libscale: Standard deviation of log-normal distribution for library sizes.
        expoutprob: Probability of a gene being an expression outlier.
        expoutloc: Mean of log-normal distribution for outlier expression.
        expoutscale: Standard deviation of log-normal distribution for outlier expression.
        ngroups: Number of cell groups/types to simulate.
        diffexpprob: Probability of a gene being differentially expressed.
        diffexpdownprob: Probability that a DE gene is downregulated.
        diffexploc: Mean of log-normal distribution for DE fold changes.
        diffexpscale: Standard deviation of log-normal distribution for DE fold changes.
        bcv_dispersion: Biological coefficient of variation dispersion.
        bcv_dof: Degrees of freedom for BCV chi-square distribution.
        ndoublets: Number of doublet cells to simulate.
        groupprob: Probability of a cell belonging to each group.
        nproggenes: Number of genes in the expression program.
        progdownprob: Probability that a program gene is downregulated.
        progdeloc: Mean of log-normal distribution for program DE.
        progdescale: Standard deviation of log-normal distribution for program DE.
        proggroups: Which cell groups can have the program active.
        progcellfrac: Fraction of cells in eligible groups with active program.
        minprogusage: Minimum program usage level.
        maxprogusage: Maximum program usage level.
    """

    ngenes: int = 10000
    ncells: int = 100
    seed: int = 757578
    mean_rate: float = 0.3
    mean_shape: float = 0.6
    libloc: float = 11.0
    libscale: float = 0.2
    expoutprob: float = 0.05
    expoutloc: float = 4.0
    expoutscale: float = 0.5
    ngroups: int = 1
    diffexpprob: float = 0.1
    diffexpdownprob: float = 0.5
    diffexploc: float = 0.1
    diffexpscale: float = 0.4
    bcv_dispersion: float = 0.1
    bcv_dof: int = 60
    ndoublets: int = 0
    groupprob: Optional[Sequence[float]] = None
    nproggenes: Optional[int] = None
    progdownprob: Optional[float] = None
    progdeloc: Optional[float] = None
    progdescale: Optional[float] = None
    proggroups: Optional[Sequence[int]] = None
    progcellfrac: Optional[float] = None
    minprogusage: float = 0.2
    maxprogusage: float = 0.8

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self._validate()

    def _validate(self) -> None:
        """Validate that all parameters are within acceptable ranges."""
        # Validate positive integers
        if self.ngenes <= 0:
            raise ValueError("ngenes must be positive")
        if self.ncells <= 0:
            raise ValueError("ncells must be positive")
        if self.ngroups <= 0:
            raise ValueError("ngroups must be positive")
        if self.bcv_dof <= 0:
            raise ValueError("bcv_dof must be positive")
        if self.ndoublets < 0:
            raise ValueError("ndoublets must be non-negative")
        if self.ndoublets > self.ncells:
            raise ValueError("ndoublets cannot exceed ncells")

        # Validate probabilities (must be between 0 and 1)
        prob_params = {
            "expoutprob": self.expoutprob,
            "diffexpprob": self.diffexpprob,
            "diffexpdownprob": self.diffexpdownprob,
        }
        for name, value in prob_params.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")

        # Validate positive floats
        if self.mean_rate <= 0:
            raise ValueError("mean_rate must be positive")
        if self.mean_shape <= 0:
            raise ValueError("mean_shape must be positive")
        if self.libscale < 0:
            raise ValueError("libscale must be non-negative")
        if self.bcv_dispersion < 0:
            raise ValueError("bcv_dispersion must be non-negative")

        # Validate program usage range
        if not 0 <= self.minprogusage <= 1:
            raise ValueError("minprogusage must be between 0 and 1")
        if not 0 <= self.maxprogusage <= 1:
            raise ValueError("maxprogusage must be between 0 and 1")
        if self.minprogusage > self.maxprogusage:
            raise ValueError("minprogusage cannot exceed maxprogusage")

        # Validate groupprob if provided
        if self.groupprob is not None:
            if len(self.groupprob) != self.ngroups:
                raise ValueError(
                    f"groupprob length ({len(self.groupprob)}) must match "
                    f"ngroups ({self.ngroups})"
                )
            if abs(sum(self.groupprob) - 1.0) > 1e-6:
                raise ValueError("groupprob must sum to 1")
            if any(p < 0 for p in self.groupprob):
                raise ValueError("groupprob values must be non-negative")

        # Validate program parameters if nproggenes is specified
        if self.nproggenes is not None and self.nproggenes > 0:
            if self.nproggenes > self.ngenes:
                raise ValueError("nproggenes cannot exceed ngenes")
            if self.progdownprob is None:
                raise ValueError("progdownprob required when nproggenes > 0")
            if not 0 <= self.progdownprob <= 1:
                raise ValueError("progdownprob must be between 0 and 1")
            if self.progdeloc is None:
                raise ValueError("progdeloc required when nproggenes > 0")
            if self.progdescale is None:
                raise ValueError("progdescale required when nproggenes > 0")
            if self.progcellfrac is None:
                raise ValueError("progcellfrac required when nproggenes > 0")
            if not 0 <= self.progcellfrac <= 1:
                raise ValueError("progcellfrac must be between 0 and 1")


@dataclass
class PerturbationConfig:
    """Configuration for perturbation effects on gene expression programs.

    The perturbation model works by replacing the activity program with a
    perturbation program. For cells with an active program:

    Control:   prog_usage × activity_program
    Perturbed: prog_usage × ((1 - response) × activity_program + response × perturb_program)

    Where `response` is sampled uniformly from [min_response, max_response] for each cell.
    Set min_response == max_response for uniform response across all cells.

    This models scenarios like CRISPR knockouts, drug treatments, or other
    interventions that modify cellular programs.

    Attributes:
        perturb_deloc: Mean of log-normal distribution for perturbation DE fold changes.
        perturb_descale: Standard deviation of log-normal for perturbation DE.
        perturb_downprob: Probability that a perturbed gene is downregulated.
        perturb_gene_frac: Fraction of program genes affected by perturbation (1.0 = all).
        min_response: Minimum perturbation response per cell (0=no effect, 1=full effect).
        max_response: Maximum perturbation response per cell (0=no effect, 1=full effect).
            Set min_response == max_response for uniform response across all cells.
        seed: Random seed for perturbation effects (separate from base simulation).
    """

    perturb_deloc: float = 0.5
    perturb_descale: float = 0.5
    perturb_downprob: float = 0.5
    perturb_gene_frac: float = 1.0
    min_response: float = 0.8
    max_response: float = 0.8
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate perturbation configuration parameters."""
        self._validate()

    def _validate(self) -> None:
        """Validate that all parameters are within acceptable ranges."""
        if not 0 <= self.perturb_downprob <= 1:
            raise ValueError("perturb_downprob must be between 0 and 1")
        if not 0 < self.perturb_gene_frac <= 1:
            raise ValueError("perturb_gene_frac must be between 0 (exclusive) and 1")
        if not 0 <= self.min_response <= 1:
            raise ValueError("min_response must be between 0 and 1")
        if not 0 <= self.max_response <= 1:
            raise ValueError("max_response must be between 0 and 1")
        if self.min_response > self.max_response:
            raise ValueError("min_response cannot exceed max_response")
