import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial import distance


@dataclass
class Source:
    """
    Create point radiation source.

    Examples
    --------
    >>> Source([1, 2, 3], 2.0)
    Source(loc=[1, 2, 3], intensity=2.0)
    """

    #: Location of the radiation source such that [x, y, z] [m].
    loc: List[float]
    #: Intensity of the radiation source [MBq]. (default is 1.0)
    intensity: float = 1.0


@dataclass
class AreaSource:
    """
    Create area radiation source. The shape is only square at this moment,
    and normal vector is along the z-axis.

    Examples
    --------
    >>> AreaSource([1, 2, 3], 2.0)
    AreaSource(bl_loc=[1, 2, 3], width=2.0, intensity=1.0, div_num=20)
    """

    #: Location of the bottom left of the radiation source such that [x, y, z] [m].
    bl_loc: List[float]
    #: Width of the square radiation area source [m].
    width: float
    #: Intensity of the radiation source [MBq]. (default is 1.0)
    intensity: float = 1.0
    #: Area source is divided into div_num*div_num point sources. (default is 20)
    div_num: int = 20


@dataclass
class Detector:
    """
    Create radiation detector. The detector only measures count.

    Examples
    --------
    >>> Detector([1, 2, 3], 2.0)
    Detector(loc=[1, 2, 3], duration=2.0, factor=5)
    """

    #: Location of the detector such that [x, y, z] [m].
    loc: List[float]
    #: Measurement time of the detector `t` [s]. (default is 10)
    duration: float = 10
    #: Factor to consider counting efficiency, size, etc. `f` (default is 5)
    #: `mu` of Poisson distribution is calculated as follows:
    #: `mu` = `ftq`/`d^2`,
    #: where `d` is the distance between the detector and sources.
    factor: float = 5


class World(object):
    """
    Create world for adding radiation sources and detectors.
    """

    def __init__(self):
        self.sources: List[Source] = []

    def __repr__(self) -> str:
        repr_str = f"{self.__class__}: contains {len(self.sources)} sources.\n"
        src_str = ",\n ".join([str(it) for it in self.sources])
        repr_str = f"{repr_str}[\n {src_str}\n]"
        return repr_str

    def add_source(self, source: Source):
        """
        Add radiation point source.

        Parameters
        ----------
        source: Source
            Radiation point source.
        """
        self.sources.append(source)

    def add_area_source(self, source: AreaSource):
        """
        Add radiation area source.
        Internally, area source is converted multiple point sources.

        Parameters
        ----------
        source: AreaSource
            Radiation area source.
        """

        # bottom left location
        bl = source.bl_loc
        q = source.intensity / source.div_num ** 2
        x_lin = np.linspace(bl[0], bl[0] + source.width, source.div_num)
        y_lin = np.linspace(bl[1], bl[1] + source.width, source.div_num)
        x_grid, y_grid = np.meshgrid(x_lin, y_lin)
        for x, y in zip(x_grid.flatten(), y_grid.flatten()):
            self.add_source(Source(loc=[x, y, bl[2]], intensity=q))

    def get_measuments(self, detectors: List[Detector]) -> np.ndarray:
        """
        Get measurement without background noise.
        Internally, area source is converted multiple point sources.

        Parameters
        ----------
        detectors: List[Detector]
            List of detectors.

        Returns
        -------
        counts: ndarray
            Counts measured by the detectors
        """
        # To list if detectors is Detector class
        if not isinstance(detectors, list):
            detectors = [detectors]

        # detectors loc
        locs = [it.loc for it in detectors]
        locs = np.stack(locs)

        # detector factors
        factors = [it.duration * it.factor for it in detectors]
        factors = np.stack(factors)

        # sources
        srcs = [it.loc for it in self.sources]
        srcs = np.stack(srcs)

        # intensity
        q = np.array([it.intensity for it in self.sources])

        dist = distance.cdist(locs, srcs)
        assert not np.any(
            dist == 0
        ), "zero division. Distance between source and detector should not be zero"

        # mu for possion distribution
        mu = factors[..., np.newaxis] * q[np.newaxis, :] / dist ** 2

        # sum of possion distribution is possion distribution
        mu = mu.sum(axis=1)

        # measument counts
        counts = stats.poisson.rvs(mu)

        return counts

    def visualize_world(
        self,
        detectors: Optional[List[Detector]] = None,
        figsize: Tuple[float, float] = (8, 8),
        plotsize: float = 10,
    ) -> plt.Axes:
        """
        Visualize radiation sources and detectors in world (X-Y plane).

        Parameters
        ----------
        detectors: Optional[List[Detector]]
            List of detectors to be visualized. (default is None)
        figsize: tuple(float, float)
            figsize for plot.
        plotsize: float
            Plot in xy plane [-plotsize, plotsize] x [-plotsize, plotsize]

        Returns
        -------
        ax: plt.Axes
            Matplotlib axes
        """

        # Setup plot area
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.plot(
            [plotsize, plotsize, -plotsize, -plotsize],
            [plotsize, -plotsize, plotsize, -plotsize],
            "wd",
        )
        ax.plot([0], [0], "rd")
        ax.set_aspect("equal")
        ax.set_title(
            "X (horizontal), Y (vertical), Origin (red dot), Detector (x) "
        )

        # Visualize sources
        x = []
        y = []
        c = []
        for it in self.sources:
            x.append(it.loc[0])
            y.append(it.loc[1])
            c.append(it.intensity)

        sc = ax.scatter(x, y, s=100, c=c, cmap=plt.cm.jet)
        _ = fig.colorbar(sc, orientation="horizontal")

        # Visualize detectors if not None
        if detectors:
            x_detec = []
            y_detec = []
            for it in detectors:
                x_detec.append(it.loc[0])
                y_detec.append(it.loc[1])
            ax.plot(x_detec, y_detec, "bx")

        return ax


if __name__ == "__main__":
    # doctest
    import doctest

    doctest.testmod()

    # Construct world
    world = World()
    world.add_source(Source(loc=[5, 8, 0], intensity=1))
    world.add_source(Source(loc=[-7, -2, 0], intensity=2.3))
    world.add_source(Source(loc=[-5.5, -2, 0], intensity=1.3))
    world.add_area_source(AreaSource(bl_loc=[5, 0, 0], width=2.0))
    ax = world.visualize_world()
    plt.show()

    # Set detectors
    div_num = 20
    x_lin = np.linspace(-10, 10, div_num)
    y_lin = np.linspace(10, -10, div_num)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)

    # Constant height
    z = 1.5
    detectors = []
    for x, y in zip(x_grid.flatten(), y_grid.flatten()):
        detectors.append(Detector(loc=[x, y, z]))

    cnts = world.get_measuments(detectors)
    print(f"max_count: {cnts.max()}")
    viz = cnts.reshape(div_num, div_num)
    plt.imshow(viz, vmin=0, vmax=np.percentile(cnts, 99))
    plt.show()
