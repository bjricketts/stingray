import warnings
from collections.abc import Generator, Iterable

import numpy as np
import matplotlib.pyplot as plt

from stingray.base import StingrayObject

from .events import EventList
from .lightcurve import Lightcurve
from .gti import cross_two_gtis
from .fourier import (
    avg_bispectrum_from_iterable,
    avg_bispectrum_from_timeseries,
    bicoherence_from_sums,
    get_flux_iterable_from_segments,
)

__all__ = ["Bispectrum", "AveragedBispectrum"]


class Bispectrum(StingrayObject):
    main_array_attr = "freq"
    type = "bispectrum"

    r"""Make a :class:`Bispectrum` from a (binned) light curve.

    The bispectrum is a higher-order spectral statistic that measures
    quadratic phase coupling between Fourier components of a time series. It is
    computed here with the direct Fourier-decomposition method (Maccarone 2013;
    Kim & Powers 1979) rather than as the Fourier transform of the third-order
    cumulant. For ``m`` averaged segments with Fourier transforms
    :math:`X_i(f)`,

    .. math::

        B(f_1, f_2) = \frac{1}{m} \sum_{i=0}^{m-1}
            X_i(f_1)\, X_i(f_2)\, X_i^{*}(f_1 + f_2)

    You can also make an empty :class:`Bispectrum` object to populate with your
    own data.

    A single :class:`Bispectrum` uses the whole light curve as one segment. To
    get a statistically meaningful bispectrum, bicoherence and biphase you
    normally want :class:`AveragedBispectrum`, which averages over many
    segments.

    Parameters
    ----------
    data : :class:`stingray.Lightcurve` or :class:`stingray.events.EventList`, optional, default ``None``
        The light curve or event list to be Fourier-transformed. If an
        :class:`EventList` is given, ``dt`` must be specified.

    Other Parameters
    ----------------
    dt : float
        The time resolution of the light curve. Only needed when the input is
        an :class:`EventList`.

    bicoherence_norm : {"kim_powers", "sigl_chamoun", "hagihira"}, default "kim_powers"
        Which normalization to use for the ``bicoherence`` attribute. All lie
        in ``[0, 1]``. With :math:`T_i = X_i(f_1) X_i(f_2) X_i^{*}(f_1+f_2)`:

        ``"kim_powers"``
            The **squared** bicoherence of Kim & Powers (1979) -- the
            plasma-physics standard (Nagashima 2006) and the form in
            Maccarone (2013):
            :math:`b^2 = |\sum_i T_i|^2 / (\sum_i |X_i(f_1)X_i(f_2)|^2 \sum_i |X_i(f_1+f_2)|^2)`.
        ``"sigl_chamoun"``
            Sigl & Chamoun (1994); the unsquared square root of the Kim &
            Powers value,
            :math:`b = |\sum_i T_i| / \sqrt{\sum_i |X_i(f_1)X_i(f_2)|^2 \sum_i |X_i(f_1+f_2)|^2}`.
        ``"hagihira"``
            Hagihira (2001) / Hayashi (2007); normalizes by the summed
            magnitude of the per-segment triple products,
            :math:`b = |\sum_i T_i| / \sum_i |T_i|`.

        Any of these can be recomputed after the fact with
        :meth:`recompute_bicoherence` (no FFTs are redone).

    poisson_subtract : bool, default False
        If True, subtract the Poisson-noise bias from each segment before
        averaging, following Wirnitzer (1985) (see Nathan et al. 2022;
        Maccarone 2013):
        :math:`X(f_1) X(f_2) X^{*}(f_1+f_2) - |X(f_1)|^2 - |X(f_2)|^2 - |X(f_1+f_2)|^2 + 2N`,
        with :math:`N` the photon count in the segment. Poisson noise biases the
        real part of the bispectrum (and hence the biphase), so this is
        recommended for photon-counting light curves given in counts. Only
        appropriate for Poisson data.

    skip_checks : bool, default False
        Skip initial checks, for speed or other reasons (you need to trust your
        inputs!).

    lc : :class:`stingray.Lightcurve`, optional
        For backwards compatibility only. Like ``data``, but no
        :class:`EventList` allowed. Deprecated.

    Attributes
    ----------
    freq : numpy.ndarray
        The array of positive Fourier frequencies that the transform samples.

    bispec : numpy.ndarray
        The complex bispectrum, an ``nf x nf`` matrix indexed by
        ``(freq, freq)``. The redundant/unresolved region (where
        ``f1 + f2`` exceeds the Nyquist frequency) is set to ``NaN``.

    bicoherence : numpy.ndarray
        The bicoherence, a real ``nf x nf`` matrix in ``[0, 1]`` (0 = no
        quadratic coupling, 1 = total coupling), computed with the
        normalization given by ``bicoherence_norm``. Note the default
        ``"kim_powers"`` returns the **squared** bicoherence. Only meaningful
        once several segments are averaged. Use :meth:`recompute_bicoherence`
        to obtain a different normalization.

    bicoherence_norm : str
        The normalization used for ``bicoherence``.

    poisson_subtracted : bool
        Whether the Poisson-noise bias was subtracted.

    biphase : numpy.ndarray
        The phase of the bispectrum, an ``nf x nf`` matrix defined over the
        full :math:`2\pi` interval.

    bispec_mag : numpy.ndarray
        Magnitude of the bispectrum, ``|bispec|``.

    bispec_phase : numpy.ndarray
        Alias of ``biphase``.

    bispec_err : numpy.ndarray
        Approximate 1-sigma uncertainty on ``bispec`` (standard error of the
        mean of the per-segment triple products). Zero for a single segment.

    biphase_err : numpy.ndarray
        Approximate 1-sigma uncertainty on ``biphase`` from circular statistics
        (Fisher 1993). Zero for a single segment.

    df : float
        The frequency resolution.

    m : int
        The number of averaged bispectra.

    n : int
        The number of data points in each segment.

    nphots : float
        The total number of photons (mean per segment).

    References
    ----------
    1) T. J. Maccarone, "The biphase explained: understanding the asymmetries
       in coupled Fourier components of astronomical time series", MNRAS 435,
       3547 (2013).

    2) Y. C. Kim and E. J. Powers, "Digital Bispectral Analysis and Its
       Applications to Nonlinear Wave Interactions", IEEE Transactions on
       Plasma Science, PS-7, 120 (1979).

    Examples
    --------
    >>> lc = Lightcurve(np.arange(64), np.random.default_rng(0).poisson(10, 64))
    >>> bs = Bispectrum(lc)
    >>> assert bs.bispec.shape[0] == bs.freq.size
    >>> assert bs.m == 1
    """

    def __init__(
        self,
        data=None,
        dt=None,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        skip_checks=False,
        lc=None,
    ):
        self._type = None
        if lc is not None:
            warnings.warn("The lc keyword is now deprecated. Use data instead", DeprecationWarning)
        if data is None:
            data = lc

        good_input = data is not None
        if good_input and not skip_checks:
            good_input = self.initial_checks(data=data, dt=dt)

        self.dt = dt
        self.gti = gti
        self.bicoherence_norm = bicoherence_norm
        self.poisson_subtracted = poisson_subtract

        if not good_input:
            return self._initialize_empty()

        return self._initialize_from_any_input(
            data,
            dt=dt,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
        )

    def initial_checks(self, data=None, dt=None, segment_size=None):
        """Run basic checks on the inputs.

        Returns ``True`` if the input can be used to build a bispectrum,
        raises otherwise. An empty (``None``) input returns ``False`` so that
        an empty object is created.
        """
        if data is None:
            return False

        if isinstance(data, EventList):
            if dt is None:
                raise ValueError(
                    "If the input is an event list, the time resolution dt " "must be specified."
                )
        elif isinstance(data, Lightcurve):
            pass
        elif isinstance(data, (tuple, list, Generator)):
            pass
        else:
            raise TypeError(f"Bad input to Bispectrum: {type(data)}")

        if segment_size is not None and dt is not None and segment_size < 2 * dt:
            raise ValueError("segment_size must be at least 2 * dt.")

        return True

    def _initialize_from_any_input(
        self,
        data,
        dt=None,
        segment_size=None,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        silent=False,
        save_all=False,
    ):
        """Initialize the object, dispatching on the type of ``data``."""
        if isinstance(data, EventList):
            spec = bispectrum_from_events(
                data,
                dt,
                segment_size=segment_size,
                gti=gti,
                bicoherence_norm=bicoherence_norm,
                poisson_subtract=poisson_subtract,
                silent=silent,
                save_all=save_all,
            )
        elif isinstance(data, Lightcurve):
            spec = bispectrum_from_lightcurve(
                data,
                segment_size=segment_size,
                gti=gti,
                bicoherence_norm=bicoherence_norm,
                poisson_subtract=poisson_subtract,
                silent=silent,
                save_all=save_all,
            )
        elif isinstance(data, (tuple, list, Generator)):
            data = list(data)
            if len(data) == 0 or not isinstance(data[0], Lightcurve):  # pragma: no cover
                raise TypeError(f"Bad inputs to Bispectrum: {type(data[0]) if data else None}")
            dt = data[0].dt
            spec = bispectrum_from_lc_iterable(
                data,
                dt,
                segment_size=segment_size,
                gti=gti,
                bicoherence_norm=bicoherence_norm,
                poisson_subtract=poisson_subtract,
                silent=silent,
                save_all=save_all,
            )
        else:  # pragma: no cover
            raise TypeError(f"Bad inputs to Bispectrum: {type(data)}")

        for key, val in spec.__dict__.items():
            setattr(self, key, val)
        return

    def _initialize_empty(self):
        """Set all attributes to ``None`` (or sensible defaults)."""
        self.freq = None
        self.bispec = None
        self.bicoherence = None
        self.bicoherence_norm = getattr(self, "bicoherence_norm", "kim_powers")
        self.poisson_subtracted = getattr(self, "poisson_subtracted", False)
        self.biphase = None
        self.bispec_mag = None
        self.bispec_phase = None
        self.bispec_err = None
        self.biphase_err = None
        self.valid = None
        self.bispec_all = None
        self._bicoh_abs_bispec_sum = None
        self._bicoh_denom1 = None
        self._bicoh_denom2 = None
        self._bicoh_sum_abs = None
        self.df = None
        self.dt = None
        self.m = 1
        self.n = None
        self.nphots = None
        self.segment_size = None
        self.gti = None
        return

    def recompute_bicoherence(self, norm, inplace=False):
        """Recompute the bicoherence under a different normalization.

        Uses the accumulated bispectrum sums stored on the object, so no FFTs
        are recomputed. See :class:`Bispectrum` for the definition of each
        normalization.

        Parameters
        ----------
        norm : {"kim_powers", "sigl_chamoun", "hagihira"}
            The bicoherence normalization.

        Other Parameters
        ----------------
        inplace : bool, default False
            If ``True``, also overwrite ``self.bicoherence`` and
            ``self.bicoherence_norm``.

        Returns
        -------
        bicoherence : numpy.ndarray
            The recomputed bicoherence.
        """
        if getattr(self, "_bicoh_denom1", None) is None:
            raise ValueError("This Bispectrum has no data to compute a bicoherence from.")
        bicoh = bicoherence_from_sums(
            norm,
            self._bicoh_abs_bispec_sum,
            self._bicoh_denom1,
            self._bicoh_denom2,
            self._bicoh_sum_abs,
            valid=self.valid,
        )
        if inplace:
            self.bicoherence = bicoh
            self.bicoherence_norm = norm.lower()
        return bicoh

    @staticmethod
    def from_lightcurve(
        lc, gti=None, bicoherence_norm="kim_powers", poisson_subtract=False, silent=False
    ):
        """Calculate a :class:`Bispectrum` from a light curve.

        Parameters
        ----------
        lc : :class:`stingray.Lightcurve`
            Light curve to be analyzed.

        Other Parameters
        ----------------
        gti : ``[[gti0_0, gti0_1], ...]``
            Good time intervals.
        bicoherence_norm : {"kim_powers", "sigl_chamoun", "hagihira"}, default "kim_powers"
            The bicoherence normalization (see :class:`Bispectrum`).
        silent : bool, default False
            Silence the progress bars.
        """
        return bispectrum_from_lightcurve(
            lc,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            silent=silent,
        )

    @staticmethod
    def from_events(
        events, dt, gti=None, bicoherence_norm="kim_powers", poisson_subtract=False, silent=False
    ):
        """Calculate a :class:`Bispectrum` from an event list.

        Parameters
        ----------
        events : :class:`stingray.EventList`
            Event list to be analyzed.
        dt : float
            The time resolution of the intermediate light curve (sets the
            Nyquist frequency).

        Other Parameters
        ----------------
        gti : ``[[gti0_0, gti0_1], ...]``
            Good time intervals.
        bicoherence_norm : {"kim_powers", "sigl_chamoun", "hagihira"}, default "kim_powers"
            The bicoherence normalization (see :class:`Bispectrum`).
        silent : bool, default False
            Silence the progress bars.
        """
        return bispectrum_from_events(
            events,
            dt,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            silent=silent,
        )

    @staticmethod
    def from_time_array(
        times, dt, gti=None, bicoherence_norm="kim_powers", poisson_subtract=False, silent=False
    ):
        """Calculate a :class:`Bispectrum` from an array of event times.

        Parameters
        ----------
        times : `np.array`
            Event arrival times.
        dt : float
            The time resolution of the intermediate light curve.

        Other Parameters
        ----------------
        gti : ``[[gti0_0, gti0_1], ...]``
            Good time intervals.
        bicoherence_norm : {"kim_powers", "sigl_chamoun", "hagihira"}, default "kim_powers"
            The bicoherence normalization (see :class:`Bispectrum`).
        silent : bool, default False
            Silence the progress bars.
        """
        return bispectrum_from_time_array(
            times,
            dt,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            silent=silent,
        )

    @staticmethod
    def from_stingray_timeseries(
        ts,
        flux_attr,
        error_flux_attr=None,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        silent=False,
    ):
        """Calculate a :class:`Bispectrum` from a time series.

        Parameters
        ----------
        ts : :class:`stingray.StingrayTimeseries`
            Input time series.
        flux_attr : str
            The attribute of the time series to use as flux.

        Other Parameters
        ----------------
        error_flux_attr : str
            The attribute of the time series to use as error bar.
        gti : ``[[gti0_0, gti0_1], ...]``
            Good time intervals.
        bicoherence_norm : {"kim_powers", "sigl_chamoun", "hagihira"}, default "kim_powers"
            The bicoherence normalization (see :class:`Bispectrum`).
        silent : bool, default False
            Silence the progress bars.
        """
        return bispectrum_from_stingray_timeseries(
            ts,
            flux_attr,
            error_flux_attr=error_flux_attr,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            silent=silent,
        )

    def plot_mag(self, ax=None, save=False, filename=None):
        """Plot the magnitude of the bispectrum as a function of frequency.

        Parameters
        ----------
        ax : ``matplotlib.axes.Axes``, default ``None``
            The axes to plot onto. A new one is created if ``None``.
        save : bool, default ``False``
            If ``True``, save the figure to ``filename``.
        filename : str, default ``None``
            File name to save the figure to. Defaults to ``bispec_mag.png``.

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            The axes with the plot.
        """
        return self._plot_matrix(
            self.bispec_mag, "Bispectrum Magnitude", ax, save, filename, "bispec_mag.png"
        )

    def plot_phase(self, ax=None, save=False, filename=None):
        """Plot the biphase as a function of frequency.

        Parameters
        ----------
        ax : ``matplotlib.axes.Axes``, default ``None``
            The axes to plot onto. A new one is created if ``None``.
        save : bool, default ``False``
            If ``True``, save the figure to ``filename``.
        filename : str, default ``None``
            File name to save the figure to. Defaults to ``bispec_phase.png``.

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            The axes with the plot.
        """
        return self._plot_matrix(self.biphase, "Biphase", ax, save, filename, "bispec_phase.png")

    def plot_bicoherence(self, ax=None, save=False, filename=None):
        """Plot the bicoherence as a function of frequency.

        Parameters
        ----------
        ax : ``matplotlib.axes.Axes``, default ``None``
            The axes to plot onto. A new one is created if ``None``.
        save : bool, default ``False``
            If ``True``, save the figure to ``filename``.
        filename : str, default ``None``
            File name to save the figure to. Defaults to ``bicoherence.png``.

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            The axes with the plot.
        """
        return self._plot_matrix(
            self.bicoherence, "Bicoherence", ax, save, filename, "bicoherence.png"
        )

    def _plot_matrix(self, matrix, title, ax, save, filename, default_filename):
        """Shared helper for the 2D bispectrum plots."""
        if matrix is None:
            raise ValueError("This Bispectrum has no data to plot.")

        if ax is None:
            _, ax = plt.subplots()

        cont = ax.contourf(self.freq, self.freq, matrix, 100, cmap=plt.cm.Spectral_r)
        ax.figure.colorbar(cont, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Frequency 1 (Hz)")
        ax.set_ylabel("Frequency 2 (Hz)")

        if save:
            ax.figure.savefig(filename if filename is not None else default_filename)
        return ax

    def plot_jellyfish(
        self,
        f0=None,
        freqs=None,
        bicoherence_levels=(0.01, 0.05),
        fundamental_color="tab:blue",
        subharmonic_color="tab:orange",
        other_color="0.6",
        ax=None,
        save=False,
        filename=None,
    ):
        r"""Draw a "jellyfish plot" of the autobispectrum (Nathan et al. 2022).

        For each frequency :math:`\nu` on the autobispectrum diagonal
        (:math:`f_1 = f_2 = \nu`, which couples :math:`\nu` and its harmonic
        :math:`2\nu`), the per-segment triple products
        :math:`X_i(\nu) X_i(\nu) X_i^{*}(2\nu)` are accumulated segment by
        segment and the running (cumulative) sum is traced as a path in the
        complex plane. Each path is normalized so that the amplitude of its
        end point equals the bicoherence (Sigl & Chamoun convention), and its
        angle is the biphase.

        Frequencies that are quadratically phase-coupled produce per-segment
        contributions that point in a consistent direction, so their path walks
        steadily outward into a "tentacle"; uncoupled frequencies random-walk
        near the origin, forming the "body". Reference circles of constant
        bicoherence give the scale.

        This requires an averaged bispectrum built with ``save_all=True`` so
        that the per-segment bispectra are available.

        Parameters
        ----------
        f0 : float, optional
            A reference (e.g. QPO fundamental) frequency to highlight. The
            diagonal path closest to ``f0`` is drawn in ``fundamental_color``
            (it couples the fundamental and its harmonic), and the path closest
            to ``f0 / 2`` in ``subharmonic_color`` (the subharmonic and the
            fundamental). If ``None``, every path is drawn in ``other_color``.

        Other Parameters
        ----------------
        freqs : iterable of float, optional
            The diagonal frequencies to draw. Defaults to every resolved
            diagonal frequency (those for which :math:`2\nu` is at or below the
            Nyquist frequency).
        bicoherence_levels : iterable of float, default ``(0.01, 0.05)``
            Radii, in bicoherence units, of the reference circles.
        fundamental_color, subharmonic_color, other_color : color
            Colors for the fundamental, subharmonic and remaining paths.
        ax : ``matplotlib.axes.Axes``, optional
            Axes to draw onto. A new one is created if ``None``.
        save : bool, default ``False``
            If ``True``, save the figure to ``filename``.
        filename : str, optional
            File name to save to. Defaults to ``bispec_jellyfish.png``.

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            The axes with the plot.

        Notes
        -----
        The bispectrum is biased by Poisson noise; this plot shows the raw
        (uncorrected) bispectrum, so paths in the low-bicoherence body carry a
        noise contribution.
        """
        if getattr(self, "bispec_all", None) is None:
            raise ValueError(
                "plot_jellyfish needs the per-segment bispectra. Build the "
                "AveragedBispectrum with save_all=True."
            )

        subbs = np.asarray(self.bispec_all)  # (m, nf, nf)
        nf = self.freq.size
        diag = np.arange(nf)
        valid_diag = diag[np.diag(self.valid)]

        if freqs is not None:
            wanted = [int(np.argmin(np.abs(self.freq - f))) for f in np.atleast_1d(freqs)]
            valid_diag = np.array([j for j in wanted if self.valid[j, j]])

        if valid_diag.size == 0:
            raise ValueError("No resolved diagonal frequencies to plot.")

        norm = np.sqrt(
            self._bicoh_denom1[valid_diag, valid_diag] * self._bicoh_denom2[valid_diag, valid_diag]
        )

        # Cumulative sum of the per-segment triples along the diagonal, starting
        # from the origin. Shape (n_freq, m + 1).
        diag_triples = subbs[:, valid_diag, valid_diag]  # (m, n_freq)
        cumsum = np.cumsum(diag_triples, axis=0)
        paths = np.vstack([np.zeros(valid_diag.size), cumsum]) / norm[np.newaxis, :]

        j_fund = j_sub = None
        if f0 is not None:
            j_fund = valid_diag[np.argmin(np.abs(self.freq[valid_diag] - f0))]
            j_sub = valid_diag[np.argmin(np.abs(self.freq[valid_diag] - f0 / 2.0))]

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))

        # Reference circles of constant bicoherence
        theta = np.linspace(0, 2 * np.pi, 200)
        for level in bicoherence_levels:
            ax.plot(
                level * np.cos(theta),
                level * np.sin(theta),
                ls="--",
                color="0.7",
                lw=1,
                zorder=1,
                label=f"bicoherence {level:g}",
            )

        # Draw the "other" paths first, highlighted ones on top
        fund_label_done = sub_label_done = False
        for col, j in enumerate(valid_diag):
            path = paths[:, col]
            if j == j_fund:
                color, lw, z = fundamental_color, 1.8, 4
                label = None if fund_label_done else "QPO fundamental"
                fund_label_done = True
            elif j == j_sub:
                color, lw, z = subharmonic_color, 1.8, 4
                label = None if sub_label_done else "subharmonic"
                sub_label_done = True
            else:
                color, lw, z = other_color, 0.8, 2
                label = None
            ax.plot(path.real, path.imag, color=color, lw=lw, alpha=0.9, zorder=z, label=label)
            ax.plot(path.real[-1], path.imag[-1], ".", color=color, ms=6, zorder=z + 1)

        lim = 1.15 * max(np.max(np.abs(paths)), max(bicoherence_levels))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="0.9", lw=0.8, zorder=0)
        ax.axvline(0, color="0.9", lw=0.8, zorder=0)
        ax.set_xlabel("Re(B)")
        ax.set_ylabel("Im(B)")
        ax.set_title("Bispectrum jellyfish plot")
        ax.legend(loc="upper right", fontsize="small")

        if save:
            ax.figure.savefig(filename if filename is not None else "bispec_jellyfish.png")
        return ax


class AveragedBispectrum(Bispectrum):
    type = "bispectrum"

    r"""Make an averaged bispectrum from a light curve or event list.

    The light curve is split into segments of length ``segment_size``, a
    bispectrum is computed for each segment, and the results are averaged (see
    :class:`Bispectrum` for the estimator definition). Averaging is what makes
    the bicoherence and biphase statistically meaningful.

    Parameters
    ----------
    data : :class:`stingray.Lightcurve`, iterable of :class:`stingray.Lightcurve`, or :class:`stingray.events.EventList`
        The light curve data to be Fourier-transformed.

    segment_size : float
        The size, in seconds, of each segment to average. If the total duration
        is not an integer multiple of ``segment_size``, the leftover at the end
        is discarded.

    Other Parameters
    ----------------
    gti : 2-d float array
        ``[[gti0_0, gti0_1], ...]`` -- Good time intervals.

    dt : float
        The time resolution of the light curve. Only needed when the input is
        an :class:`EventList`.

    silent : bool, default False
        Do not show a progress bar.

    save_all : bool, default False
        Save all intermediate bispectra used for the final average (under
        ``bispec_all``). Use with care; this can fill up RAM.

    skip_checks : bool, default False
        Skip initial checks, for speed or other reasons (you need to trust your
        inputs!).

    lc : :class:`stingray.Lightcurve`, optional
        For backwards compatibility only. Deprecated; use ``data``.

    Attributes
    ----------
    See :class:`Bispectrum`. In addition:

    segment_size : float
        The size of each averaged segment.

    bispec_all : list of numpy.ndarray
        Only present if ``save_all=True``: the per-segment bispectra.
    """

    def __init__(
        self,
        data=None,
        segment_size=None,
        gti=None,
        dt=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        silent=False,
        save_all=False,
        skip_checks=False,
        lc=None,
    ):
        self._type = None
        if lc is not None:
            warnings.warn("The lc keyword is now deprecated. Use data instead", DeprecationWarning)
        if data is None:
            data = lc

        good_input = data is not None
        if good_input and not skip_checks:
            good_input = self.initial_checks(data=data, dt=dt, segment_size=segment_size)

        self.dt = dt
        self.gti = gti
        self.bicoherence_norm = bicoherence_norm
        self.poisson_subtracted = poisson_subtract
        self.segment_size = segment_size
        self.save_all = save_all

        if not good_input:
            return self._initialize_empty()

        if isinstance(data, Generator):
            warnings.warn(
                "The averaged bispectrum from a generator of light curves "
                "pre-allocates the full list of light curves, losing the "
                "advantage of lazy loading. If that matters to you, use the "
                "AveragedBispectrum.from_lc_iterable static method, specifying "
                "the sampling time dt."
            )
            data = list(data)

        return self._initialize_from_any_input(
            data,
            dt=dt,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            silent=silent,
            save_all=save_all,
        )

    def initial_checks(self, data=None, dt=None, segment_size=None):
        if data is not None and segment_size is None:
            raise ValueError("segment_size must be specified for an AveragedBispectrum.")
        return super().initial_checks(data=data, dt=dt, segment_size=segment_size)

    @staticmethod
    def from_lightcurve(
        lc,
        segment_size,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        silent=False,
        save_all=False,
    ):
        """Calculate an :class:`AveragedBispectrum` from a light curve."""
        return bispectrum_from_lightcurve(
            lc,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            silent=silent,
            save_all=save_all,
        )

    @staticmethod
    def from_events(
        events,
        dt,
        segment_size,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        silent=False,
        save_all=False,
    ):
        """Calculate an :class:`AveragedBispectrum` from an event list."""
        return bispectrum_from_events(
            events,
            dt,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            silent=silent,
            save_all=save_all,
        )

    @staticmethod
    def from_time_array(
        times,
        dt,
        segment_size,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        silent=False,
        save_all=False,
    ):
        """Calculate an :class:`AveragedBispectrum` from an array of event times."""
        return bispectrum_from_time_array(
            times,
            dt,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            silent=silent,
            save_all=save_all,
        )

    @staticmethod
    def from_stingray_timeseries(
        ts,
        flux_attr,
        segment_size,
        error_flux_attr=None,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        silent=False,
        save_all=False,
    ):
        """Calculate an :class:`AveragedBispectrum` from a time series."""
        return bispectrum_from_stingray_timeseries(
            ts,
            flux_attr,
            error_flux_attr=error_flux_attr,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            silent=silent,
            save_all=save_all,
        )

    @staticmethod
    def from_lc_iterable(
        iter_lc,
        dt,
        segment_size,
        gti=None,
        bicoherence_norm="kim_powers",
        poisson_subtract=False,
        silent=False,
        save_all=False,
    ):
        """Calculate an :class:`AveragedBispectrum` from an iterable of light curves."""
        return bispectrum_from_lc_iterable(
            iter_lc,
            dt,
            segment_size=segment_size,
            gti=gti,
            bicoherence_norm=bicoherence_norm,
            poisson_subtract=poisson_subtract,
            silent=silent,
            save_all=save_all,
        )


def _create_bispectrum_from_result_table(table, force_averaged=False):
    """Populate a :class:`Bispectrum` or :class:`AveragedBispectrum` from a
    result table produced by ``stingray.fourier.avg_bispectrum_from_XX``.
    """
    if table is None:  # pragma: no cover
        raise ValueError("No usable segments were found to compute the bispectrum.")

    if table.meta["m"] > 1 or force_averaged:
        bs = AveragedBispectrum()
    else:
        bs = Bispectrum()

    bs.freq = np.asarray(table.meta["freq"])
    bs.bispec = table.meta["bispec"]
    bs.bicoherence = table.meta["bicoherence"]
    bs.bicoherence_norm = table.meta["bicoherence_norm"]
    bs.poisson_subtracted = table.meta.get("poisson_subtract", False)
    bs.biphase = table.meta["biphase"]
    bs.bispec_err = table.meta["bispec_err"]
    bs.biphase_err = table.meta["biphase_err"]
    bs.valid = table.meta["valid"]
    bs.bispec_mag = np.abs(bs.bispec)
    bs.bispec_phase = bs.biphase

    # Raw accumulated sums, kept so the bicoherence can be recomputed under a
    # different normalization via ``recompute_bicoherence`` without redoing FFTs.
    bs._bicoh_abs_bispec_sum = table.meta["bicoh_abs_bispec_sum"]
    bs._bicoh_denom1 = table.meta["bicoh_denom1"]
    bs._bicoh_denom2 = table.meta["bicoh_denom2"]
    bs._bicoh_sum_abs = table.meta["bicoh_sum_abs"]

    for attr in ["n", "m", "dt", "df", "nphots", "segment_size", "gti"]:
        if attr in table.meta:
            setattr(bs, attr, table.meta[attr])

    if "subbs" in table.meta:
        bs.bispec_all = table.meta["subbs"]

    return bs


def bispectrum_from_time_array(
    times,
    dt,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    silent=False,
    save_all=False,
):
    """Calculate a bispectrum from an array of event times.

    Parameters
    ----------
    times : `np.array`
        Event arrival times.
    dt : float
        The time resolution of the intermediate light curves.

    Other Parameters
    ----------------
    segment_size : float
        The length, in seconds, of the light curve segments to average. Only
        relevant (and required) for an :class:`AveragedBispectrum`.
    gti : ``[[gti0_0, gti0_1], ...]``
        Good time intervals.
    bicoherence_norm : {"kim_powers", "sigl_chamoun", "hagihira"}, default "kim_powers"
        The bicoherence normalization (see :class:`Bispectrum`).
    silent : bool, default False
        Silence the progress bars.
    save_all : bool, default False
        Save all intermediate bispectra used for the final average.

    Returns
    -------
    spec : :class:`AveragedBispectrum` or :class:`Bispectrum`
        The output bispectrum.
    """
    force_averaged = segment_size is not None
    silent = silent or (segment_size is None)
    table = avg_bispectrum_from_timeseries(
        times,
        gti,
        segment_size,
        dt,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        silent=silent,
        return_subbs=save_all,
    )
    return _create_bispectrum_from_result_table(table, force_averaged=force_averaged)


def bispectrum_from_events(
    events,
    dt,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    silent=False,
    save_all=False,
):
    """Calculate a bispectrum from an event list. See
    `bispectrum_from_time_array` for the parameters."""
    if gti is None:
        gti = events.gti
    dt = events.suggest_compatible_dt(dt)
    return bispectrum_from_time_array(
        events.time,
        dt,
        segment_size=segment_size,
        gti=gti,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        silent=silent,
        save_all=save_all,
    )


def bispectrum_from_lightcurve(
    lc,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    silent=False,
    save_all=False,
):
    """Calculate a bispectrum from a light curve. See
    `bispectrum_from_time_array` for the parameters."""
    force_averaged = segment_size is not None
    silent = silent or (segment_size is None)
    if gti is None:
        gti = lc.gti
    err = None
    if lc.err_dist == "gauss":
        err = lc.counts_err
    table = avg_bispectrum_from_timeseries(
        lc.time,
        gti,
        segment_size,
        lc.dt,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        silent=silent,
        fluxes=lc.counts,
        errors=err,
        return_subbs=save_all,
    )
    return _create_bispectrum_from_result_table(table, force_averaged=force_averaged)


def bispectrum_from_stingray_timeseries(
    ts,
    flux_attr,
    error_flux_attr=None,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    silent=False,
    save_all=False,
):
    """Calculate a bispectrum from a time series. See
    `bispectrum_from_time_array` for the parameters."""
    force_averaged = segment_size is not None
    silent = silent or (segment_size is None)
    if gti is None:
        gti = ts.gti
    err = None
    if error_flux_attr is not None:
        err = getattr(ts, error_flux_attr)
    table = avg_bispectrum_from_timeseries(
        ts.time,
        gti,
        segment_size,
        ts.dt,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        silent=silent,
        fluxes=getattr(ts, flux_attr),
        errors=err,
        return_subbs=save_all,
    )
    return _create_bispectrum_from_result_table(table, force_averaged=force_averaged)


def bispectrum_from_lc_iterable(
    iter_lc,
    dt,
    segment_size=None,
    gti=None,
    bicoherence_norm="kim_powers",
    poisson_subtract=False,
    silent=False,
    save_all=False,
):
    """Calculate an average bispectrum from an iterable of light curves.

    Parameters
    ----------
    iter_lc : iterable of :class:`stingray.Lightcurve` or `np.array`
        Light curves. If arrays, they are used as counts.
    dt : float
        The time resolution of the light curves.

    Other Parameters
    ----------------
    segment_size : float, default None
        The length, in seconds, of the light curve segments to average.
    gti : ``[[gti0_0, gti0_1], ...]``
        Good time intervals.
    silent : bool, default False
        Silence the progress bars.
    save_all : bool, default False
        Save all intermediate bispectra used for the final average.

    Returns
    -------
    spec : :class:`AveragedBispectrum` or :class:`Bispectrum`
        The output bispectrum.
    """
    force_averaged = segment_size is not None
    silent = silent or (segment_size is None)
    common_gti = gti

    def iterate_lc_counts(iter_lc):
        for lc in iter_lc:
            if hasattr(lc, "counts"):
                n_bin = (
                    np.rint(segment_size / lc.dt).astype(int) if segment_size else lc.counts.size
                )
                lc_gti = lc.gti
                if common_gti is not None:
                    lc_gti = cross_two_gtis(common_gti, lc.gti)
                err = None
                if lc.err_dist == "gauss":
                    err = lc.counts_err
                flux_iterable = get_flux_iterable_from_segments(
                    lc.time, lc_gti, segment_size, n_bin, fluxes=lc.counts, errors=err
                )
                for out in flux_iterable:
                    yield out
            elif isinstance(lc, Iterable):
                yield lc
            else:
                raise TypeError(
                    "The inputs to bispectrum_from_lc_iterable must be "
                    "Lightcurve objects or arrays."
                )

    table = avg_bispectrum_from_iterable(
        iterate_lc_counts(iter_lc),
        dt,
        bicoherence_norm=bicoherence_norm,
        poisson_subtract=poisson_subtract,
        silent=silent,
        return_subbs=save_all,
    )
    return _create_bispectrum_from_result_table(table, force_averaged=force_averaged)
