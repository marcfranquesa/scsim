"""Single-cell RNA-seq simulator using the Splatter statistical framework."""

import logging
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd
from numpy.random import Generator

from .config import SimulationConfig

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
        logger.info("Simulating cells")
        self.cellparams = self._get_cell_params()

        logger.info("Simulating gene params")
        self.geneparams = self._get_gene_params()

        nproggenes = self.config.nproggenes
        if nproggenes is not None and nproggenes > 0:
            logger.info("Simulating program")
            self._simulate_program()

        logger.info("Simulating DE")
        self._sim_group_de()

        logger.info("Simulating cell-gene means")
        self.cellgenemean = self._get_cell_gene_means()

        if self.config.ndoublets > 0:
            logger.info("Simulating doublets")
            self._simulate_doublets()

        logger.info("Adjusting means")
        self._adjust_means_bcv()

        logger.info("Simulating counts")
        self._simulate_counts()

        return self

    def _simulate_counts(self) -> None:
        """Sample read counts from Poisson distribution.

        Uses the variance-trend adjusted mean values to generate
        integer count data.
        """
        self.counts = pd.DataFrame(
            self._rng.poisson(lam=self.updatedmean),
            index=self._cellnames,
            columns=self._genenames,
        )

    def _adjust_means_bcv(self) -> None:
        """Adjust cell-gene means to follow a mean-variance trend.

        Applies biological coefficient of variation to create
        realistic overdispersion in the count data.
        """
        self.bcv = self.config.bcv_dispersion + (1 / np.sqrt(self.cellgenemean))
        chisamp = self._rng.chisquare(self.config.bcv_dof, size=self.config.ngenes)
        self.bcv = self.bcv * np.sqrt(self.config.bcv_dof / chisamp)
        self.updatedmean = self._rng.gamma(
            shape=1 / (self.bcv**2), scale=self.cellgenemean * (self.bcv**2)
        )
        self.bcv = pd.DataFrame(
            self.bcv, index=self._cellnames, columns=self._genenames
        )
        self.updatedmean = pd.DataFrame(
            self.updatedmean, index=self._cellnames, columns=self._genenames
        )

    def _simulate_doublets(self) -> None:
        """Simulate doublet cells by merging expression profiles.

        Doublets are created by combining expression profiles from
        two cells while preserving the total library size.
        """
        ncells = self.config.ncells
        ndoublets = self.config.ndoublets

        # Select doublet cells and determine the second cell to merge with
        d_ind = sorted(self._rng.choice(ncells, ndoublets, replace=False))
        d_ind = [f"Cell{x + 1}" for x in d_ind]
        self.cellparams["is_doublet"] = False
        self.cellparams.loc[d_ind, "is_doublet"] = True
        extraind = self.cellparams.index[-ndoublets:]
        group2 = self.cellparams.loc[extraind, "group"].values
        self.cellparams["group2"] = -1
        self.cellparams.loc[d_ind, "group2"] = group2

        # Update the cell-gene means for the doublets while preserving
        # the same library size
        dmean = self.cellgenemean.loc[d_ind, :].values
        dmultiplier = 0.5 / dmean.sum(axis=1)
        dmean = np.multiply(dmean, dmultiplier[:, np.newaxis])
        omean = self.cellgenemean.loc[extraind, :].values
        omultiplier = 0.5 / omean.sum(axis=1)
        omean = np.multiply(omean, omultiplier[:, np.newaxis])
        newmean = dmean + omean
        libsize = self.cellparams.loc[d_ind, "libsize"].values
        newmean = np.multiply(newmean, libsize[:, np.newaxis])
        self.cellgenemean.loc[d_ind, :] = newmean

        # Remove extra doublet cells from the data structures
        self.cellgenemean.drop(extraind, axis=0, inplace=True)
        self.cellparams.drop(extraind, axis=0, inplace=True)
        self._cellnames = self._cellnames[:ncells]

    def _get_cell_gene_means(self) -> pd.DataFrame:
        """Calculate each gene's mean expression for each cell.

        Combines group-specific expression with program effects and
        normalizes by library size.

        Returns:
            DataFrame with cells as rows and genes as columns containing
            expected expression values.
        """
        group_genemean = self.geneparams.loc[
            :,
            [
                x
                for x in self.geneparams.columns
                if ("_genemean" in x) and ("group" in x)
            ],
        ].T.astype(float)
        group_genemean = group_genemean.div(group_genemean.sum(axis=1), axis=0)
        ind = self.cellparams["group"].apply(lambda x: f"group{x}_genemean")

        nproggenes = self.config.nproggenes
        if nproggenes is None or nproggenes == 0:
            cellgenemean = group_genemean.loc[ind, :].astype(float)
            cellgenemean.index = self.cellparams.index
        else:
            noprogcells = ~self.cellparams["has_program"]
            hasprogcells = self.cellparams["has_program"]

            logger.debug("Getting mean for activity program carrying cells")
            progcellmean = group_genemean.loc[ind[hasprogcells], :]
            progcellmean.index = ind.index[hasprogcells]
            progcellmean = progcellmean.multiply(
                1 - self.cellparams.loc[hasprogcells, "program_usage"], axis=0
            )

            progmean = self.geneparams.loc[:, ["prog_genemean"]]
            progmean = progmean.div(progmean.sum(axis=0), axis=1)
            progusage = self.cellparams.loc[progcellmean.index, ["program_usage"]]
            progusage.columns = ["prog_genemean"]
            progcellmean += progusage.dot(progmean.T)
            progcellmean = progcellmean.astype(float)

            logger.debug("Getting mean for non activity program carrying cells")
            noprogcellmean = group_genemean.loc[ind[noprogcells], :]
            noprogcellmean.index = ind.index[noprogcells]

            cellgenemean = pd.concat([noprogcellmean, progcellmean], axis=0)
            cellgenemean = cellgenemean.reindex(index=self.cellparams.index)

        logger.debug("Normalizing by cell libsize")
        normfac = (self.cellparams["libsize"] / cellgenemean.sum(axis=1)).values
        cellgenemean = cellgenemean.multiply(normfac, axis=0)
        return cellgenemean

    def _get_gene_params(self) -> pd.DataFrame:
        """Sample gene expression parameters.

        Generates base expression levels from a gamma distribution and
        identifies outlier genes with higher expression.

        Returns:
            DataFrame with gene parameters including base mean, outlier
            status, and adjusted gene mean.
        """
        cfg = self.config
        ngenes = cfg.ngenes

        basegenemean = self._rng.gamma(
            shape=cfg.mean_shape, scale=1.0 / cfg.mean_rate, size=ngenes
        )

        is_outlier = self._rng.choice(
            [True, False], size=ngenes, p=[cfg.expoutprob, 1 - cfg.expoutprob]
        )
        outlier_ratio = np.ones(shape=ngenes)
        outliers = self._rng.lognormal(
            mean=cfg.expoutloc, sigma=cfg.expoutscale, size=is_outlier.sum()
        )
        outlier_ratio[is_outlier] = outliers
        gene_mean = basegenemean.copy()
        median = np.median(basegenemean)
        gene_mean[is_outlier] = outliers * median
        self._genenames = [f"Gene{i}" for i in range(1, ngenes + 1)]
        geneparams = pd.DataFrame(
            [basegenemean, is_outlier, outlier_ratio, gene_mean],
            index=["BaseGeneMean", "is_outlier", "outlier_ratio", "gene_mean"],
            columns=self._genenames,
        ).T
        return geneparams

    def _get_cell_params(self) -> pd.DataFrame:
        """Sample cell group identities and library sizes.

        Returns:
            DataFrame with cell parameters including group assignment
            and library size.
        """
        cfg = self.config
        groupid = self._simulate_groups()
        libsize = self._rng.lognormal(
            mean=cfg.libloc, sigma=cfg.libscale, size=self._init_ncells
        )
        self._cellnames = [f"Cell{i}" for i in range(1, self._init_ncells + 1)]
        cellparams = pd.DataFrame(
            [groupid, libsize], index=["group", "libsize"], columns=self._cellnames
        ).T
        cellparams["group"] = cellparams["group"].astype(int)
        return cellparams

    def _simulate_program(self) -> None:
        """Simulate a shared gene expression program.

        Creates a program affecting a subset of genes that is active
        in a fraction of cells with varying usage levels.
        """
        cfg = self.config
        nproggenes = cfg.nproggenes
        ngenes = cfg.ngenes

        # Simulate the program gene expression
        self.geneparams["prog_gene"] = False
        proggenes = self.geneparams.index[-nproggenes:]
        self.geneparams.loc[proggenes, "prog_gene"] = True
        de_ratio = self._rng.lognormal(
            mean=cfg.progdeloc, sigma=cfg.progdescale, size=nproggenes
        )
        de_ratio[de_ratio < 1] = 1 / de_ratio[de_ratio < 1]
        is_downregulated = self._rng.choice(
            [True, False],
            size=len(de_ratio),
            p=[cfg.progdownprob, 1 - cfg.progdownprob],
        )
        de_ratio[is_downregulated] = 1.0 / de_ratio[is_downregulated]
        all_de_ratio = np.ones(ngenes)
        all_de_ratio[-nproggenes:] = de_ratio
        prog_mean = self.geneparams["gene_mean"] * all_de_ratio
        self.geneparams["prog_genemean"] = prog_mean

        # Assign the program to cells
        self.cellparams["has_program"] = False
        proggroups = cfg.proggroups
        if proggroups is None:
            # The program is active in all cell types
            proggroups = np.arange(1, cfg.ngroups + 1)

        self.cellparams.loc[:, "program_usage"] = 0.0  # Use float to avoid dtype warning
        for g in proggroups:
            groupcells = self.cellparams.index[self.cellparams["group"] == g]
            hasprog = self._rng.choice(
                [True, False],
                size=len(groupcells),
                p=[cfg.progcellfrac, 1 - cfg.progcellfrac],
            )
            self.cellparams.loc[groupcells[hasprog], "has_program"] = True
            usages = self._rng.uniform(
                low=cfg.minprogusage,
                high=cfg.maxprogusage,
                size=len(groupcells[hasprog]),
            )
            self.cellparams.loc[groupcells[hasprog], "program_usage"] = usages

    def _simulate_groups(self) -> np.ndarray:
        """Sample cell group identities from a categorical distribution.

        Returns:
            Array of group assignments for each cell.
        """
        groupid = self._rng.choice(
            np.arange(1, self.config.ngroups + 1),
            size=self._init_ncells,
            p=self._groupprob,
        )
        self._groups = np.unique(groupid)
        return groupid

    def _sim_group_de(self) -> None:
        """Simulate differential expression between cell groups.

        For each group, randomly selects DE genes and assigns
        fold changes from a log-normal distribution.
        """
        cfg = self.config
        ngenes = cfg.ngenes
        nproggenes = cfg.nproggenes

        if nproggenes is not None and nproggenes > 0:
            proggene = self.geneparams["prog_gene"].values
        else:
            proggene = np.array([False] * self.geneparams.shape[0])

        for group in self._groups:
            is_de = self._rng.choice(
                [True, False],
                size=ngenes,
                p=[cfg.diffexpprob, 1 - cfg.diffexpprob],
            )
            # Program genes shouldn't be differentially expressed between groups
            is_de[proggene] = False

            de_ratio = self._rng.lognormal(
                mean=cfg.diffexploc, sigma=cfg.diffexpscale, size=is_de.sum()
            )
            de_ratio[de_ratio < 1] = 1 / de_ratio[de_ratio < 1]
            is_downregulated = self._rng.choice(
                [True, False],
                size=len(de_ratio),
                p=[cfg.diffexpdownprob, 1 - cfg.diffexpdownprob],
            )
            de_ratio[is_downregulated] = 1.0 / de_ratio[is_downregulated]
            all_de_ratio = np.ones(ngenes)
            all_de_ratio[is_de] = de_ratio
            group_mean = self.geneparams["gene_mean"] * all_de_ratio

            deratiocol = f"group{group}_DEratio"
            groupmeancol = f"group{group}_genemean"
            self.geneparams[deratiocol] = all_de_ratio
            self.geneparams[groupmeancol] = group_mean

    def to_anndata(self) -> "anndata.AnnData":
        """Export simulation results to an AnnData object.

        Requires the `anndata` package to be installed.
        Install with: `pip install scsim[anndata]`

        Returns:
            AnnData object with:
            - X: count matrix (cells x genes)
            - obs: cell metadata from cellparams
            - var: gene metadata from geneparams
            - uns["scsim_config"]: simulation config as dict

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

        from dataclasses import asdict

        # Create AnnData object
        adata = anndata.AnnData(
            X=self.counts.values,
            obs=self.cellparams.copy(),
            var=self.geneparams.copy(),
        )

        # Store config in uns
        adata.uns["scsim_config"] = asdict(self.config)

        return adata
