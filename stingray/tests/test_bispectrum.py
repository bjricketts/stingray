import os
import copy

import numpy as np
import pytest
import matplotlib.pyplot as plt

from stingray import Lightcurve, EventList, StingrayTimeseries
from stingray.bispectrum import Bispectrum, AveragedBispectrum
from stingray.fourier import (
    fftfreq,
    positive_fft_bins,
    avg_bispectrum_from_iterable,
    avg_bispectrum_from_timeseries,
    bicoherence_from_sums,
    BICOHERENCE_NORMS,
    _bispectrum_frequency_grid,
)


def clear_all_figs():
    fign = plt.get_fignums()
    for fig in fign:
        plt.close(fig)


rng = np.random.RandomState(20150907)

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


def _brute_force_bispectrum(segments, n_bin):
    """Reference bispectrum and the three bicoherence normalizations.

    A direct, slow, double-loop implementation of the Maccarone (2013)
    estimator, used to validate the vectorized ``avg_bispectrum_*`` functions.
    """
    fgt0 = positive_fft_bins(n_bin)
    kbins = np.arange(fgt0.start, fgt0.stop)
    nf = len(kbins)
    nyq = n_bin // 2
    num = np.zeros((nf, nf), dtype=complex)
    den1 = np.zeros((nf, nf))
    den2 = np.zeros((nf, nf))
    sum_abs = np.zeros((nf, nf))
    valid = np.zeros((nf, nf), dtype=bool)
    for s in segments:
        ft = np.fft.fft(np.asarray(s))
        for a, ka in enumerate(kbins):
            for b, kb in enumerate(kbins):
                if ka + kb > nyq:
                    continue
                valid[a, b] = True
                t = ft[ka] * ft[kb] * np.conj(ft[ka + kb])
                num[a, b] += t
                den1[a, b] += np.abs(ft[ka] * ft[kb]) ** 2
                den2[a, b] += np.abs(ft[ka + kb]) ** 2
                sum_abs[a, b] += np.abs(t)
    k = len(segments)
    bispec = np.full((nf, nf), np.nan, dtype=complex)
    norms = {n: np.full((nf, nf), np.nan) for n in BICOHERENCE_NORMS}
    bispec[valid] = num[valid] / k
    norms["kim_powers"][valid] = np.abs(num[valid]) ** 2 / (den1[valid] * den2[valid])
    norms["sigl_chamoun"][valid] = np.abs(num[valid]) / np.sqrt(den1[valid] * den2[valid])
    norms["hagihira"][valid] = np.abs(num[valid]) / sum_abs[valid]
    return bispec, norms, valid


def _coupled_segments(rng_local, n_seg, n_bin, dt, f1, f2, couple=True):
    """Segments with (optionally) quadratic phase coupling of f1, f2 -> f1+f2."""
    t = np.arange(n_bin) * dt
    segs = []
    for _ in range(n_seg):
        p1, p2 = rng_local.uniform(0, 2 * np.pi, size=2)
        p3 = p1 + p2 if couple else rng_local.uniform(0, 2 * np.pi)
        s = (
            np.cos(2 * np.pi * f1 * t + p1)
            + np.cos(2 * np.pi * f2 * t + p2)
            + np.cos(2 * np.pi * (f1 + f2) * t + p3)
            + 20
        )
        segs.append(s)
    return segs


class TestBispectrum(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 0.1
        cls.n = 256
        cls.time = np.arange(cls.n) * cls.dt
        cls.counts = rng.poisson(50, cls.n).astype(float)
        cls.lc = Lightcurve(cls.time, cls.counts, dt=cls.dt, skip_checks=True)
        cls.events = EventList(
            np.sort(rng.uniform(0, cls.n * cls.dt, 5000)), gti=[[0, cls.n * cls.dt]]
        )
        cls.bs = Bispectrum(cls.lc)

    @pytest.mark.parametrize("skip_checks", [True, False])
    def test_initialize_empty(self, skip_checks):
        bs = Bispectrum(skip_checks=skip_checks)
        assert bs.freq is None
        assert bs.bispec is None
        assert bs.m == 1

    def test_make_empty_bispectrum(self):
        bs = Bispectrum()
        assert bs.bicoherence is None
        assert bs.biphase is None
        assert bs.bispec_mag is None

    def test_make_bispectrum_from_lightcurve(self):
        bs = Bispectrum(self.lc)
        assert isinstance(bs, Bispectrum)
        assert bs.m == 1
        assert bs.n == self.n

    def test_bispectrum_types(self):
        assert self.bs.type == "bispectrum"

    def test_type_change(self):
        bs = copy.deepcopy(self.bs)
        assert bs.type == "bispectrum"
        bs.type = "astdfawerfsaf"
        assert bs.type == "astdfawerfsaf"

    def test_shapes(self):
        nf = self.bs.freq.size
        assert self.bs.bispec.shape == (nf, nf)
        assert self.bs.bicoherence.shape == (nf, nf)
        assert self.bs.biphase.shape == (nf, nf)
        assert self.bs.bispec_mag.shape == (nf, nf)
        assert self.bs.bispec_err.shape == (nf, nf)

    def test_init_without_lightcurve(self):
        with pytest.raises(TypeError):
            Bispectrum(self.lc.counts)

    def test_init_with_nonsense_data(self):
        nonsense_data = [None for i in range(100)]
        with pytest.raises(TypeError):
            Bispectrum(nonsense_data)

    def test_init_with_wrong_data_type(self):
        with pytest.raises(TypeError):
            Bispectrum(1)

    def test_eventlist_needs_dt(self):
        with pytest.raises(ValueError):
            Bispectrum(self.events)

    def test_lc_keyword_deprecation(self):
        with pytest.warns(DeprecationWarning) as record:
            bs1 = Bispectrum(lc=self.lc)
        assert np.any(["deprecated" in r.message.args[0].lower() for r in record])
        bs2 = Bispectrum(data=self.lc)
        good = np.isfinite(bs1.bispec)
        assert np.allclose(bs1.bispec[good], bs2.bispec[good])

    def test_bispec_mag_and_phase(self):
        good = np.isfinite(self.bs.bispec)
        assert np.allclose(self.bs.bispec_mag[good], np.abs(self.bs.bispec[good]))
        assert np.allclose(self.bs.bispec_phase[good], self.bs.biphase[good])

    def test_redundant_region_is_nan(self):
        assert np.any(np.isnan(self.bs.bispec))
        assert np.array_equal(np.isnan(self.bs.bispec), ~self.bs.valid)

    def test_from_lightcurve_works(self):
        bs = Bispectrum.from_lightcurve(self.lc)
        good = np.isfinite(bs.bispec)
        assert np.allclose(bs.bispec[good], self.bs.bispec[good])

    def test_from_events_works(self):
        bs = Bispectrum.from_events(self.events, dt=self.dt)
        assert isinstance(bs, Bispectrum)

    def test_from_time_array_works(self):
        bs = Bispectrum.from_time_array(self.events.time, dt=self.dt, gti=[[0, self.n * self.dt]])
        assert isinstance(bs, Bispectrum)

    def test_from_stingray_timeseries_works(self):
        ts = StingrayTimeseries(
            self.time, array_attrs={"flux": self.counts}, dt=self.dt, skip_checks=True
        )
        ts.gti = self.lc.gti
        bs = Bispectrum.from_stingray_timeseries(ts, "flux")
        good = np.isfinite(bs.bispec)
        assert np.allclose(bs.bispec[good], self.bs.bispec[good])


class TestAveragedBispectrum(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 0.1
        cls.n = 512
        cls.segment_size = 5.0
        cls.time = np.arange(cls.n) * cls.dt
        cls.counts = rng.poisson(40, cls.n).astype(float)
        cls.lc = Lightcurve(cls.time, cls.counts, dt=cls.dt, skip_checks=True)
        cls.events = EventList(
            np.sort(rng.uniform(0, cls.n * cls.dt, 8000)), gti=[[0, cls.n * cls.dt]]
        )
        cls.bs = AveragedBispectrum(cls.lc, segment_size=cls.segment_size)

    @pytest.mark.parametrize("skip_checks", [True, False])
    def test_initialize_empty(self, skip_checks):
        bs = AveragedBispectrum(skip_checks=skip_checks)
        assert bs.freq is None
        assert bs.m == 1

    def test_averaged_bispectrum_from_lightcurve(self):
        assert isinstance(self.bs, AveragedBispectrum)
        assert self.bs.m > 1

    @pytest.mark.parametrize("nseg", [1, 2, 5, 10])
    def test_n_segments(self, nseg):
        segment_size = self.n * self.dt / nseg
        bs = AveragedBispectrum(self.lc, segment_size=segment_size)
        assert bs.m == nseg

    def test_segments_with_leftover(self):
        # A segment size that does not divide the light curve evenly
        bs = AveragedBispectrum(self.lc, segment_size=self.segment_size * 1.5)
        assert bs.m > 1

    def test_init_without_segment(self):
        with pytest.raises(ValueError):
            AveragedBispectrum(self.lc)

    def test_init_with_none_segment(self):
        with pytest.raises(ValueError):
            AveragedBispectrum(self.lc, segment_size=None)

    def test_lc_keyword_deprecation(self):
        with pytest.warns(DeprecationWarning):
            AveragedBispectrum(lc=self.lc, segment_size=self.segment_size)

    def test_from_lightcurve_works(self):
        bs = AveragedBispectrum.from_lightcurve(self.lc, segment_size=self.segment_size)
        good = np.isfinite(bs.bispec)
        assert np.allclose(bs.bispec[good], self.bs.bispec[good])

    def test_from_events_works(self):
        bs = AveragedBispectrum.from_events(self.events, dt=self.dt, segment_size=self.segment_size)
        assert isinstance(bs, AveragedBispectrum)
        assert bs.m > 1

    def test_from_time_array_works(self):
        bs = AveragedBispectrum.from_time_array(
            self.events.time,
            dt=self.dt,
            segment_size=self.segment_size,
            gti=[[0, self.n * self.dt]],
        )
        assert bs.m > 1

    def test_list_of_light_curves(self):
        bs = AveragedBispectrum([self.lc, self.lc], segment_size=self.segment_size)
        assert isinstance(bs, AveragedBispectrum)

    def test_from_lc_iterable_works(self):
        bs = AveragedBispectrum.from_lc_iterable(
            [self.lc, self.lc], self.dt, segment_size=self.segment_size
        )
        assert isinstance(bs, AveragedBispectrum)

    def test_list_with_nonsense_component(self):
        with pytest.raises(TypeError):
            AveragedBispectrum([self.lc, 3], segment_size=self.segment_size)

    def test_save_all(self):
        bs = AveragedBispectrum.from_lightcurve(
            self.lc, segment_size=self.segment_size, save_all=True
        )
        assert hasattr(bs, "bispec_all")
        assert len(bs.bispec_all) == bs.m

    def test_gti_is_threaded_through(self):
        full = AveragedBispectrum(self.lc, segment_size=self.segment_size)
        half = AveragedBispectrum(
            self.lc, segment_size=self.segment_size, gti=[[0, self.n * self.dt / 2]]
        )
        assert half.m < full.m

    def test_skip_checks(self):
        AveragedBispectrum(self.lc, segment_size=self.segment_size, skip_checks=True)


class TestBispectrumEstimator(object):
    """Tests of the underlying Fourier estimator functions in ``fourier.py``
    (``avg_bispectrum_from_iterable``, ``avg_bispectrum_from_timeseries``,
    ``bicoherence_from_sums``, ``_bispectrum_frequency_grid``).
    """

    @classmethod
    def setup_class(cls):
        cls.dt = 0.1
        cls.n_bin = 32
        cls.segments = [rng.poisson(25, cls.n_bin).astype(float) for _ in range(40)]

        # For the counts-vs-events equality test
        cls.length = 100.0
        cls.ctrate = 1000
        cls.segment_size = 5.0
        cls.N = int(cls.length / cls.dt)
        cls.times = np.sort(rng.uniform(0, cls.length, int(cls.length * cls.ctrate)))
        cls.gti = np.asanyarray([[0, cls.length]])
        cls.counts, bins = np.histogram(cls.times, bins=np.linspace(0, cls.length, cls.N + 1))
        cls.bin_times = (bins[:-1] + bins[1:]) / 2

    def test_frequency_grid(self):
        freq, idx1, idx2, idx3, valid = _bispectrum_frequency_grid(self.n_bin, self.dt)
        fgt0 = positive_fft_bins(self.n_bin)
        assert np.allclose(freq, fftfreq(self.n_bin, self.dt)[fgt0])
        assert idx1.shape == (freq.size, freq.size)
        # The valid region is where f1 + f2 is at or below the Nyquist bin
        assert np.all(idx3[valid] <= self.n_bin // 2)

    def test_no_segments_returns_none(self):
        assert avg_bispectrum_from_iterable(iter([]), self.dt, silent=True) is None

    def test_matches_brute_force(self):
        res = avg_bispectrum_from_iterable(iter(self.segments), self.dt, silent=True)
        bispec_ref, norms_ref, valid = _brute_force_bispectrum(self.segments, self.n_bin)
        assert np.array_equal(res.meta["valid"], valid)
        assert np.allclose(res.meta["bispec"][valid], bispec_ref[valid], atol=1e-9)
        assert np.allclose(
            res.meta["bicoherence"][valid], norms_ref["kim_powers"][valid], atol=1e-12
        )

    @pytest.mark.parametrize("norm", BICOHERENCE_NORMS)
    def test_all_norms_match_brute_force(self, norm):
        _, norms_ref, valid = _brute_force_bispectrum(self.segments, self.n_bin)
        res = avg_bispectrum_from_iterable(
            iter(self.segments), self.dt, bicoherence_norm=norm, silent=True
        )
        assert np.allclose(res.meta["bicoherence"][valid], norms_ref[norm][valid], atol=1e-12)

    def test_kim_powers_is_sigl_chamoun_squared(self):
        rk = avg_bispectrum_from_iterable(
            iter(self.segments), self.dt, bicoherence_norm="kim_powers", silent=True
        )
        rs = avg_bispectrum_from_iterable(
            iter(self.segments), self.dt, bicoherence_norm="sigl_chamoun", silent=True
        )
        valid = rk.meta["valid"]
        assert np.allclose(
            rk.meta["bicoherence"][valid], rs.meta["bicoherence"][valid] ** 2, atol=1e-12
        )

    def test_biphase_is_angle_of_bispectrum(self):
        res = avg_bispectrum_from_iterable(iter(self.segments), self.dt, silent=True)
        valid = res.meta["valid"]
        d = np.angle(
            np.exp(1j * (res.meta["biphase"][valid] - np.angle(res.meta["bispec"][valid])))
        )
        assert np.allclose(d, 0, atol=1e-9)

    def test_single_segment_bicoherence_is_one(self):
        res = avg_bispectrum_from_iterable(iter(self.segments[:1]), self.dt, silent=True)
        valid = res.meta["valid"]
        assert np.allclose(res.meta["bicoherence"][valid], 1.0)
        assert np.allclose(res.meta["biphase_err"][valid], 0.0)
        assert res.meta["m"] == 1

    def test_bad_norm_raises(self):
        with pytest.raises(ValueError):
            avg_bispectrum_from_iterable(
                iter(self.segments), self.dt, bicoherence_norm="bogus", silent=True
            )

    def test_bicoherence_from_sums_bad_norm(self):
        ones = np.ones((2, 2))
        with pytest.raises(ValueError):
            bicoherence_from_sums("bogus", ones, ones, ones, ones)

    def test_cts_and_events_are_equal(self):
        bs_evts = avg_bispectrum_from_timeseries(
            self.times, self.gti, self.segment_size, self.dt, silent=True
        )
        bs_cts = avg_bispectrum_from_timeseries(
            self.bin_times, self.gti, self.segment_size, self.dt, fluxes=self.counts, silent=True
        )
        valid = bs_evts.meta["valid"]
        assert np.allclose(bs_evts.meta["bispec"][valid], bs_cts.meta["bispec"][valid])
        assert np.allclose(bs_evts.meta["bicoherence"][valid], bs_cts.meta["bicoherence"][valid])

    @pytest.mark.parametrize("norm", BICOHERENCE_NORMS)
    def test_detects_quadratic_coupling(self, norm):
        rng_local = np.random.RandomState(99)
        dt, n_bin = 0.01, 128
        f1, f2 = 5.0, 12.0
        coupled = _coupled_segments(rng_local, 300, n_bin, dt, f1, f2, couple=True)
        control = _coupled_segments(rng_local, 300, n_bin, dt, f1, f2, couple=False)
        res_c = avg_bispectrum_from_iterable(iter(coupled), dt, bicoherence_norm=norm, silent=True)
        res_u = avg_bispectrum_from_iterable(iter(control), dt, bicoherence_norm=norm, silent=True)
        freq = res_c.meta["freq"]
        i1 = np.argmin(np.abs(freq - f1))
        i2 = np.argmin(np.abs(freq - f2))
        # Strong coupling gives a high bicoherence; the control (same power
        # spectrum, random relative phase) gives a low one, for every norm.
        assert res_c.meta["bicoherence"][i1, i2] > 0.7
        assert res_u.meta["bicoherence"][i1, i2] < 0.4
        assert res_c.meta["bicoherence"][i1, i2] > res_u.meta["bicoherence"][i1, i2]

    def test_biphase_of_coupling(self):
        rng_local = np.random.RandomState(5)
        dt, n_bin = 0.01, 128
        f1, f2 = 5.0, 12.0
        segs = _coupled_segments(rng_local, 200, n_bin, dt, f1, f2, couple=True)
        res = avg_bispectrum_from_iterable(iter(segs), dt, silent=True)
        freq = res.meta["freq"]
        i1 = np.argmin(np.abs(freq - f1))
        i2 = np.argmin(np.abs(freq - f2))
        assert abs(res.meta["biphase"][i1, i2]) < 0.2


class TestBispectrumNormalization(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 0.01
        n = 256 * 20
        cls.lc = Lightcurve(
            np.arange(n) * cls.dt, rng.poisson(60, n).astype(float), dt=cls.dt, skip_checks=True
        )
        cls.segment_size = 2.56

    def test_default_is_kim_powers(self):
        bs = AveragedBispectrum(self.lc, segment_size=self.segment_size)
        assert bs.bicoherence_norm == "kim_powers"

    @pytest.mark.parametrize("norm", ["kim_powers", "sigl_chamoun", "hagihira"])
    def test_norm_parameter(self, norm):
        bs = AveragedBispectrum(self.lc, segment_size=self.segment_size, bicoherence_norm=norm)
        assert bs.bicoherence_norm == norm
        valid = bs.valid
        assert np.all(bs.bicoherence[valid] >= 0)
        assert np.all(bs.bicoherence[valid] <= 1)

    @pytest.mark.parametrize("norm", ["kim_powers", "sigl_chamoun", "hagihira"])
    def test_recompute_matches_direct(self, norm):
        bs = AveragedBispectrum(self.lc, segment_size=self.segment_size)
        direct = AveragedBispectrum(self.lc, segment_size=self.segment_size, bicoherence_norm=norm)
        recomputed = bs.recompute_bicoherence(norm)
        valid = bs.valid
        assert np.allclose(recomputed[valid], direct.bicoherence[valid], atol=1e-12)

    def test_recompute_inplace(self):
        bs = AveragedBispectrum(self.lc, segment_size=self.segment_size)
        bs.recompute_bicoherence("hagihira", inplace=True)
        assert bs.bicoherence_norm == "hagihira"

    def test_recompute_bad_norm(self):
        bs = AveragedBispectrum(self.lc, segment_size=self.segment_size)
        with pytest.raises(ValueError):
            bs.recompute_bicoherence("nope")

    def test_recompute_on_empty_raises(self):
        with pytest.raises(ValueError):
            Bispectrum().recompute_bicoherence("kim_powers")


class TestBispectrumIO(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 0.1
        cls.n = 256
        lc = Lightcurve(
            np.arange(cls.n) * cls.dt,
            rng.poisson(40, cls.n).astype(float),
            dt=cls.dt,
            skip_checks=True,
        )
        cls.bs = AveragedBispectrum(lc, segment_size=5.0)

    def test_astropy_table_roundtrip(self):
        ts = self.bs.to_astropy_table()
        back = AveragedBispectrum.from_astropy_table(ts)
        assert np.allclose(back.freq, self.bs.freq)
        assert np.allclose(np.nan_to_num(back.bispec), np.nan_to_num(self.bs.bispec))
        assert np.allclose(np.nan_to_num(back.bicoherence), np.nan_to_num(self.bs.bicoherence))

    def test_recompute_survives_roundtrip(self):
        ts = self.bs.to_astropy_table()
        back = AveragedBispectrum.from_astropy_table(ts)
        valid = self.bs.valid
        assert np.allclose(
            back.recompute_bicoherence("sigl_chamoun")[valid],
            self.bs.recompute_bicoherence("sigl_chamoun")[valid],
            atol=1e-12,
        )

    @pytest.mark.parametrize("fmt", ["pickle"])
    def test_file_roundtrip(self, fmt):
        fname = f"test_bispec.{fmt}"
        try:
            self.bs.write(fname, fmt=fmt)
            back = AveragedBispectrum.read(fname, fmt=fmt)
            assert np.allclose(back.freq, self.bs.freq)
            assert np.allclose(np.nan_to_num(back.bispec), np.nan_to_num(self.bs.bispec))
        finally:
            if os.path.exists(fname):
                os.remove(fname)


class TestBispectrumPlots(object):
    @classmethod
    def setup_class(cls):
        lc = Lightcurve(
            np.arange(256) * 0.1, rng.poisson(40, 256).astype(float), dt=0.1, skip_checks=True
        )
        cls.bs = AveragedBispectrum(lc, segment_size=5.0)

    def teardown_method(self):
        clear_all_figs()

    def test_plot_mag(self):
        ax = self.bs.plot_mag()
        assert ax is not None

    def test_plot_phase(self):
        ax = self.bs.plot_phase()
        assert ax is not None

    def test_plot_bicoherence(self):
        ax = self.bs.plot_bicoherence()
        assert ax is not None

    def test_plot_on_given_axis(self):
        _, ax = plt.subplots()
        out = self.bs.plot_mag(ax=ax)
        assert out is ax

    def test_plot_save(self, tmp_path):
        fname = str(tmp_path / "mag.png")
        self.bs.plot_mag(save=True, filename=fname)
        assert os.path.exists(fname)

    def test_plot_empty_raises(self):
        with pytest.raises(ValueError):
            Bispectrum().plot_mag()
