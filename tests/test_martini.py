"""Test the functionality of the main martini module."""

import re
import itertools
import pytest
import warnings
import numpy as np
from martini.martini import Martini, GlobalProfile, _BaseMartini, _plan_segments
from martini.datacube import DataCube, HIfreq
from martini._grid_search import FindGridIntersectionsResult
import martini._util
import martini.martini
from martini._util import resolve_ncpu
from martini.beams import GaussianBeam
from test_sph_kernels import simple_kernels, recommended_kernels
from test_beams import OffsetBeam
from martini.sph_kernels import _CubicSplineKernel, _GaussianKernel, DiracDeltaKernel
from martini.spectral_models import DiracDeltaSpectrum, GaussianSpectrum
from martini.sources import SPHSource, CombinedSource
from astropy import units as U
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import FK5, ICRS, SkyCoord
from scipy.signal import fftconvolve
from scipy.spatial.transform import Rotation


def check_mass_accuracy(m, out_mode, out_dir):
    """
    Check that mass in cube matches input mass by better than 1%.

    Parameters
    ----------
    m : ~martini.martini.Martini
        The martini instance to test for accuracy.

    out_mode : str
        Can be ``"fits"`` or ``"hdf5"``.

    out_dir : ~pathlib.Path
        Directory to write the output file to.
    """
    if out_mode == "hdf5":
        h5py = pytest.importorskip(
            "h5py", reason="h5py (optional dependency) not available"
        )

    # flux in channels
    F = (m.datacube._array * m.datacube.px_size**2).sum((0, 1)).squeeze()  # Jy

    # distance
    D = m.source.distance

    # channel width
    dv = np.abs(np.diff(m.datacube.velocity_channel_edges))

    # HI mass
    MHI = np.sum(
        2.36e5
        * U.Msun
        * D.to_value(U.Mpc) ** 2
        * F.to_value(U.Jy)
        * dv.to_value(U.km / U.s)
    ).to(U.Msun)

    # demand accuracy within 1% after source insertion
    assert U.isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

    m.convolve_beam()

    # radiant intensity
    Irad = m.datacube._array.sum((0, 1)).squeeze()  # Jy / beam

    # beam area, for an equivalent Gaussian beam
    A = np.pi * m.beam.bmaj * m.beam.bmin / 4 / np.log(2) / U.beam

    # distance
    D = m.source.distance

    # channel width
    dv = np.abs(np.diff(m.datacube.velocity_channel_edges))

    # flux
    F = (Irad / A).to(U.Jy / U.arcsec**2) * m.datacube.px_size**2

    # HI mass
    MHI = np.sum(
        2.36e5
        * U.Msun
        * D.to_value(U.Mpc) ** 2
        * F.to_value(U.Jy)
        * dv.to_value(U.km / U.s)
    ).to(U.Msun)

    # demand accuracy within 1% after beam convolution
    assert U.isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

    if out_mode == "fits":
        filename = str(out_dir / "cube.fits")
        m.write_fits(filename)
        with fits.open(filename) as f:
            # distance
            D = m.source.distance

            # radiant intensity
            fits_wcs = wcs.WCS(f[0].header)
            Irad = U.Quantity(
                f[0].data.T.sum((0, 1)).squeeze(), unit=f[0].header["BUNIT"]
            )

            A = (
                np.pi
                * (f[0].header["BMAJ"] * U.deg)
                * (f[0].header["BMIN"] * U.deg)
                / 4
                / np.log(2)
                / U.beam
            )
            px_area = U.Quantity(
                np.abs(f[0].header["CDELT1"]), unit=f[0].header["CUNIT1"]
            ) * U.Quantity(np.abs(f[0].header["CDELT2"]), unit=f[0].header["CUNIT2"])

            # flux
            F = (Irad / A).to(U.Jy / U.arcsec**2) * px_area

            channel_edges = fits_wcs.sub(("spectral",)).all_pix2world(
                np.arange(fits_wcs.sub(("spectral",)).pixel_shape[0] + 1) - 0.5,
                0,
            ) * U.Unit(fits_wcs.wcs.cunit[fits_wcs.wcs.spec], format="fits")
            dv = np.abs(
                np.diff(
                    channel_edges.squeeze().to(
                        U.km / U.s, equivalencies=U.doppler_radio(HIfreq)
                    )
                )
            )

            # HI mass
            MHI = np.sum(
                2.36e5
                * U.Msun
                * D.to_value(U.Mpc) ** 2
                * F.to_value(U.Jy)
                * dv.to_value(U.km / U.s)
            ).to(U.Msun)

        # demand accuracy within 1% in output fits file
        assert U.isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

    if out_mode == "hdf5":
        filename = str(out_dir / "cube.hdf5")
        m.write_hdf5(filename)
        with h5py.File(filename, "r") as f:
            # distance
            D = m.source.distance

            # radiant intensity
            Irad = U.Quantity(
                f["FluxCube"][()].sum((0, 1)).squeeze(),
                unit=f["FluxCube"].attrs["FluxCubeUnit"],
            )

            A = (
                np.pi
                * (f["FluxCube"].attrs["BeamMajor_in_deg"] * U.deg)
                * (f["FluxCube"].attrs["BeamMinor_in_deg"] * U.deg)
                / 4
                / np.log(2)
                / U.beam
            )
            dv = np.abs(
                np.diff(
                    f["velocity_channel_edges"]
                    * U.Unit(f["velocity_channel_edges"].attrs["Unit"])
                )
            )
            px_area = U.Quantity(
                np.abs(f["FluxCube"].attrs["deltaRA_in_RAUnit"]),
                unit=f["FluxCube"].attrs["RAUnit"],
            ) * U.Quantity(
                np.abs(f["FluxCube"].attrs["deltaDec_in_DecUnit"]),
                unit=f["FluxCube"].attrs["DecUnit"],
            )

            # flux
            F = (Irad / A).to(U.Jy / U.arcsec**2) * px_area

            # HI mass
            MHI = np.sum(
                2.36e5
                * U.Msun
                * D.to_value(U.Mpc) ** 2
                * F.to_value(U.Jy)
                * dv.to_value(U.km / U.s)
            ).to(U.Msun)

        # demand accuracy within 1% in output hdf5 file
        assert U.isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)


class TestMartini:
    """Test functionality of the main martini class."""

    @pytest.mark.parametrize("sph_kernel", simple_kernels)
    @pytest.mark.parametrize("spectral_model", (DiracDeltaSpectrum, GaussianSpectrum))
    @pytest.mark.parametrize("out_mode", ("fits", "hdf5"))
    @pytest.mark.parametrize("combined_source", (True, False))
    def test_mass_accuracy(
        self,
        dc_zeros,
        sph_kernel,
        spectral_model,
        single_particle_source,
        out_mode,
        combined_source,
        tmp_path_factory,
    ):
        """
        Check that input mass in particles ends up in the output cube.

        Check that the input mass in the particles gives the correct total mass in the
        datacube, by checking the conversion back to total mass. Covers testing
        Martini.insert_source_in_cube.
        """
        hsm_g = (
            0.1 * U.kpc if sph_kernel.__name__ == "DiracDeltaKernel" else 1.0 * U.kpc
        )
        source = single_particle_source(hsm_g=hsm_g)
        if combined_source:
            source = CombinedSource([source, source])
        # single_particle_source has a mass of 1E4Msun, temperature of 1E4K
        m = Martini(
            source=source,
            datacube=dc_zeros,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=spectral_model(),
            sph_kernel=sph_kernel(),
        )
        m.insert_source_in_cube(progressbar=False)
        check_mass_accuracy(m, out_mode, tmp_path_factory.mktemp("cube"))

    @pytest.mark.parametrize("sph_kernel", simple_kernels)
    @pytest.mark.parametrize("spectral_model", (GaussianSpectrum,))
    @pytest.mark.parametrize("dtype", (None, np.float32, np.float64, np.int32))
    def test_fits_dtype(
        self,
        dc_zeros,
        sph_kernel,
        dtype,
        spectral_model,
        single_particle_source,
        tmp_path_factory,
    ):
        """Check that output file dtype matches requested dtype in write call."""
        source = single_particle_source()
        m = Martini(
            source=source,
            datacube=dc_zeros,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=spectral_model(),
            sph_kernel=sph_kernel(),
        )
        filename = str(tmp_path_factory.mktemp("cube") / "cube.fits")
        m.write_fits(filename, dtype=dtype)
        with fits.open(filename) as f:
            if dtype is None:
                dtype = np.float64
            assert f[0].data.dtype.type == dtype

    @pytest.mark.parametrize("dtype", (None, np.float32, np.int32, np.uint8, np.uint16))
    def test_fits_streamed_matches_astropy(
        self, dc_random, dtype, single_particle_source, tmp_path, monkeypatch
    ):
        """Check that streamed FITS output is identical to writing without streaming."""
        source = single_particle_source()
        m = Martini(
            source=source,
            datacube=dc_random,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=GaussianSpectrum(),
            sph_kernel=_CubicSplineKernel(),
            quiet=True,
        )
        m.write_fits(tmp_path / "streamed.fits", dtype=dtype)
        # without native FITS types, everything is written without streaming:
        monkeypatch.setattr(martini.martini, "_FITS_BITPIX", {})
        m.write_fits(tmp_path / "astropy.fits", dtype=dtype)
        diff = fits.FITSDiff(
            str(tmp_path / "streamed.fits"),
            str(tmp_path / "astropy.fits"),
            ignore_keywords=["DATE-OBS", "MJD-OBS"],
        )
        assert diff.identical, diff.report()

    def test_fits_overwrite(self, m_init, tmp_path):
        """Check that existing FITS files are only replaced if requested."""
        filename = tmp_path / "cube.fits"
        m_init.write_fits(filename)
        m_init.write_fits(filename, overwrite=True)
        with fits.open(filename) as f:
            assert len(f) == 1  # replaced, not appended to
        with pytest.raises(OSError, match="To overwrite set"):
            m_init.write_fits(filename, overwrite=False)

    def test_convolve_beam(self, single_particle_source):
        """Check that beam convolution gives result matching manual calculation."""
        source = single_particle_source()
        datacube = DataCube(
            n_px_x=16,
            n_px_y=16,
            n_channels=16,
            spectral_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
            px_size=15 * U.arcsec,
            channel_width=4 * U.km / U.s,
        )
        beam = GaussianBeam()
        noise = None
        sph_kernel = _GaussianKernel()
        spectral_model = GaussianSpectrum()

        m = Martini(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=noise,
            sph_kernel=sph_kernel,
            spectral_model=spectral_model,
        )
        m.insert_source_in_cube()
        unconvolved_cube = m.datacube._array.copy()
        unit = unconvolved_cube.unit
        s = np.s_[..., 0] if m.datacube.stokes_axis else np.s_[...]
        for spatial_slice in iter(unconvolved_cube[s].transpose((2, 0, 1))):
            spatial_slice[...] = (
                fftconvolve(spatial_slice, m.beam.kernel, mode="same") * unit
            )
        convolved_cube = unconvolved_cube[
            m.datacube.padx : -m.datacube.padx, m.datacube.pady : -m.datacube.padx
        ]
        convolved_cube = convolved_cube.to(
            U.Jy * U.beam**-1,
            equivalencies=U.beam_angular_area(m.beam.area),
        )
        m.convolve_beam()
        assert U.allclose(m.datacube._array, convolved_cube)

    def test_no_beam(self, single_particle_source):
        """Check that convolving without a beam fails."""
        source = single_particle_source()
        datacube = DataCube(
            n_px_x=16,
            n_px_y=16,
            n_channels=16,
            spectral_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
            px_size=15 * U.arcsec,
            channel_width=4 * U.km / U.s,
        )
        beam = None
        noise = None
        sph_kernel = _GaussianKernel()
        spectral_model = GaussianSpectrum()

        m = Martini(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=noise,
            sph_kernel=sph_kernel,
            spectral_model=spectral_model,
        )
        m.insert_source_in_cube()
        with pytest.warns(
            UserWarning,
            match="Skipping beam convolution, no beam object provided to Martini.",
        ):
            m.convolve_beam()

    def test_add_noise(self, m_init):
        """Check that noise provided goes into the datacube when we call add_noise."""
        m_init._datacube._allocate_cube()
        assert (m_init.datacube._array.sum() == 0).all()
        assert m_init.noise.seed is not None
        expected_noise = m_init.noise.generate(m_init.datacube, m_init.beam)
        m_init.noise.reset_rng()
        m_init.add_noise()
        # noise is generated in the units of the data cube:
        assert expected_noise.unit == m_init.datacube._array.unit
        assert U.allclose(m_init.datacube._array, expected_noise)

    def test_add_noise_without_noise(self, m_init):
        """Check that adding noise is skipped (with a warning) without a noise model."""
        m_init.noise = None
        with pytest.warns(UserWarning, match="Skipping noise, no noise object"):
            m_init.add_noise()
        assert m_init.datacube._array.sum() == 0

    def test_add_noise_without_beam(self, m_init):
        """Check that adding noise is skipped (with a warning) without a beam."""
        m_init.beam = None
        with pytest.warns(UserWarning, match="Skipping noise, no beam object"):
            m_init.add_noise()
        assert m_init.datacube._array.sum() == 0

    def test_convolve_beam_insufficient_padding(self, m_init):
        """Check that we refuse to convolve if the padding is too small for the beam."""
        m_init._datacube.drop_pad()
        with pytest.raises(ValueError, match="datacube padding insufficient"):
            m_init.convolve_beam()

    def test_write_fits_long_obj_name(self, m_init, tmp_path):
        """Check that an object name too long for the FITS header is truncated."""
        filename = tmp_path / "cube.fits"
        with pytest.warns(RuntimeWarning, match="obj_name longer than 16 characters"):
            m_init.write_fits(filename, obj_name="a_very_long_object_name")
        with fits.open(filename) as f:
            assert f[0].header["OBJECT"] == "a_very_long_obje"

    def test_write_beam_fits_without_beam(self, m_init, tmp_path):
        """Check that writing the beam fails if there is no beam."""
        m_init.beam = None
        with pytest.raises(ValueError, match="Called with beam set to 'None'"):
            m_init.write_beam_fits(tmp_path / "beam.fits")

    def test_write_hdf5_uninitialized_beam(self, m_init, tmp_path):
        """Check that writing hdf5 fails if the beam kernel is uninitialized."""
        pytest.importorskip("h5py", reason="h5py (optional dependency) not available.")
        m_init.beam.kernel = None
        with pytest.raises(ValueError, match="beam kernel uninitialized"):
            m_init.write_hdf5(str(tmp_path / "cube.hdf5"))

    def test_write_hdf5_memmap(self, m_init, tmp_path):
        """Check that writing hdf5 with ``memmap=True`` returns the open file."""
        h5py = pytest.importorskip(
            "h5py", reason="h5py (optional dependency) not available."
        )
        f = m_init.write_hdf5(str(tmp_path / "cube.hdf5"), memmap=True)
        try:
            assert isinstance(f, h5py.File)
            assert f["FluxCube"].shape == m_init.datacube._array.shape
        finally:
            f.close()

    @pytest.mark.parametrize("spatial", (True, False))
    @pytest.mark.parametrize("spectral", (True, False))
    @pytest.mark.parametrize("mass", (True, False))
    def test_prune_particles(self, many_particle_source, spatial, spectral, mass):
        """
        Test pruning particles unused in the creation of the mock.

        Check that particles offset by specific sets of (RA, Dec, v) are inside/outside
        the cube as expected. All combinations of offsets are tested at once, with one
        particle per combination.
        """
        # (offset, expected inside) along each axis
        ra_offs = ((0, True), (3, True), (9, False), (-3, True), (-9, False))  # arcsec
        dec_offs = ((0, True), (3, True), (9, False), (-3, True), (-9, False))  # arcsec
        v_offs = ((0, True), (3, True), (7, False), (-3, True), (-7, False))  # km/s
        mass_offs = ((0, False), (1, True))
        combinations = list(itertools.product(ra_offs, dec_offs, v_offs, mass_offs))
        # one (offsets, expected inside) pair of arrays per axis, one entry per particle
        (ra_off, ra_in), (dec_off, dec_in), (v_off, v_in), (mass_off, mass_in) = (
            (
                np.array([off for off, _ in axis]),
                np.array([inside for _, inside in axis]),
            )
            for axis in zip(*combinations)
        )
        expect_particle = np.ones(len(combinations), dtype=bool)
        if spatial:
            expect_particle &= ra_in & dec_in
        if spectral:
            expect_particle &= v_in
        if mass:
            expect_particle &= mass_in
        # set distance so that 1kpc = 1arcsec, then for a source at (RA, Dec) = (0, 0)
        # y and z offsets in kpc are RA and Dec offsets in arcsec, and vx is along the
        # line of sight (x = 0 so that the Hubble flow doesn't contribute)
        distance = (1 * U.kpc / 1 / U.arcsec).to(U.Mpc, U.dimensionless_angles())
        source = many_particle_source(
            distance=distance,
            ra=0 * U.deg,
            dec=0 * U.deg,
            xyz_g=np.column_stack((np.zeros(ra_off.size), ra_off, dec_off)) * U.kpc,
            vxyz_g=np.column_stack((v_off, np.zeros((v_off.size, 2)))) * U.km / U.s,
            mHI_g=mass_off * 1.0e4 * U.Msun,
            # label particles with their index to find which are kept
            T_g=(1.0e4 + np.arange(ra_off.size)) * U.K,
            hsm_g=np.ones(ra_off.size) * U.kpc,
        )
        datacube = DataCube(
            n_px_x=2,
            n_px_y=2,
            n_channels=2,
            spectral_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
            px_size=1 * U.arcsec,
            channel_width=1 * U.km / U.s,
            ra=0 * U.deg,
            dec=0 * U.deg,
        )
        # pad size will be 5, so datacube is 12x12 pixels
        beam = GaussianBeam(bmaj=1 * U.arcsec, bmin=1 * U.arcsec, truncate=4)
        sph_kernel = _CubicSplineKernel()
        spectral_model = GaussianSpectrum(sigma=1 * U.km / U.s)
        # need to use _BaseMartini below to manipulate _prune_kwargs
        # if more than 1px (datacube) + 5px (pad) + 2px (sm_range) then expect to prune
        # if more than 1px (datacube) + 4px (4*spectrum_half_width) then expect to prune
        m = _BaseMartini(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=None,
            sph_kernel=sph_kernel,
            spectral_model=spectral_model,
            _prune_kwargs={"spatial": spatial, "spectral": spectral, "mass": mass},
        )
        kept = np.rint(m.source.T_g.to_value(U.K) - 1.0e4).astype(int)
        assert set(kept) == set(np.flatnonzero(expect_particle))

    @pytest.mark.parametrize(
        ("los_off", "los_in"),
        (
            (0 * U.kpc, True),
            (3 * U.kpc, True),
            (5 * U.kpc, False),
            (9 * U.kpc, False),
            (-3 * U.kpc, True),
            (-5 * U.kpc, False),
            (-9 * U.kpc, False),
        ),
    )
    @pytest.mark.parametrize("los_distance", (True, False))
    def test_prune_los_distance(
        self, los_off, los_in, los_distance, single_particle_source
    ):
        """
        Test pruning particles along the line-of-sight distance axis.

        Check that a particle offset by a specific line-of-sight distance is
        inside/outside the cube as expected. The line-of-sight extent of the cube is
        independent of its (padded) extent in RA and Dec.
        """

        class _LOSDistanceMartini(_BaseMartini):
            _n_px_z = 6

        expect_particle = los_in if los_distance else True
        # set distance so that 1kpc = 1arcsec, pixels are locally cubic so this is also
        # (approximately) 1px along the line of sight
        distance = (1 * U.kpc / 1 / U.arcsec).to(U.Mpc, U.dimensionless_angles())
        source = single_particle_source(
            distance=distance,
            xyz_g=U.Quantity([[los_off.to_value(U.kpc), 0, 0]], U.kpc),
        )
        datacube = DataCube(
            n_px_x=2,
            n_px_y=2,
            n_channels=2,
            spectral_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
            px_size=1 * U.arcsec,
            channel_width=1 * U.km / U.s,
            ra=0 * U.deg,
            dec=0 * U.deg,
        )
        # pad size will be 5, so datacube is 12x12 pixels
        beam = GaussianBeam(bmaj=1 * U.arcsec, bmin=1 * U.arcsec, truncate=4)
        kwargs = {
            "source": source,
            "datacube": datacube,
            "beam": beam,
            "noise": None,
            "sph_kernel": _CubicSplineKernel(),
            "spectral_model": GaussianSpectrum(sigma=1 * U.km / U.s),
            "_prune_kwargs": {
                "spatial": True,
                "spectral": False,
                "mass": False,
                "los_distance": los_distance,
            },
        }
        # if more than 3px (datacube) + ~1.4px (sm_range) then expect to prune
        if not expect_particle:
            with pytest.raises(
                RuntimeError,
                match="No non-zero mHI source particles in target region.",
            ):
                _LOSDistanceMartini(**kwargs)
        else:
            assert _LOSDistanceMartini(**kwargs).source.npart == 1

    def test_prune_nothing_preserves_shape(self, many_particle_source):
        """Check that pruning with no active conditions leaves array shapes unchanged."""
        source = many_particle_source()
        npart = source.npart
        datacube = DataCube(
            n_px_x=2,
            n_px_y=2,
            n_channels=2,
            spectral_centre=source.vsys,
            px_size=1 * U.arcsec,
            channel_width=1 * U.km / U.s,
            ra=0 * U.deg,
            dec=0 * U.deg,
        )
        m = _BaseMartini(
            source=source,
            datacube=datacube,
            sph_kernel=_CubicSplineKernel(),
            spectral_model=GaussianSpectrum(),
            _prune_kwargs={"spatial": False, "spectral": False, "mass": False},
        )
        assert m.source.npart == npart
        assert m.source.mHI_g.shape == (npart,)
        assert m.source.pixcoords.shape == (3, npart)
        assert m.sph_kernel.sm_ranges.shape == (npart,)

    def test_prune_los_distance_requires_n_px_z(self, single_particle_source):
        """Check that pruning in line-of-sight distance needs a line-of-sight depth."""
        source = single_particle_source()
        datacube = DataCube(
            n_px_x=2,
            n_px_y=2,
            n_channels=2,
            spectral_centre=source.vsys,
            px_size=1 * U.arcsec,
            channel_width=1 * U.km / U.s,
            ra=0 * U.deg,
            dec=0 * U.deg,
        )
        with pytest.raises(ValueError, match="requires '_n_px_z'"):
            _BaseMartini(
                source=source,
                datacube=datacube,
                sph_kernel=_CubicSplineKernel(),
                spectral_model=GaussianSpectrum(),
                _prune_kwargs={"los_distance": True},
            )

    def test_prune_nan(self, single_particle_source):
        """
        Check that particles with nan coordinates are removed.

        Particles can have nan pixel coordinates if they're more than 90deg in longitude
        away from the reference pixel location of the WCS (pretty sure that's the
        condition, else maybe 90deg longitude away from the 0 pixel). We want these to
        be rejected in pruning if spatial pruning is active.

        The source has two particles, once translated to 10 Mpc distance they are at
        +10 and -10 Mpc diametrically opposed w.r.t. the observer. If we tested with just
        one invalid particle Martini exits complaining that there are no particles in the
        target region, so we put one valid and one invalid and check that only one
        remains after pruning.
        """
        source = SPHSource(
            xyz_g=U.Quantity([[0, -20], [0, 0], [0, 0]], U.Mpc),
            vxyz_g=U.Quantity([[0, 0], [0, 0], [0, 0]], U.km / U.s),
            T_g=U.Quantity([1e4, 1e4], U.K),
            hsm_g=U.Quantity([0.1, 0.1], U.kpc),
            mHI_g=U.Quantity([1e6, 1e6], U.Msun),
            ra=0 * U.deg,
            dec=0 * U.deg,
            distance=10 * U.Mpc,
        )
        datacube = DataCube(
            n_px_x=2,
            n_px_y=2,
            n_channels=2,
            channel_width=1 * U.km / U.s,
            px_size=10 * U.arcsec,
            spectral_centre=source.vsys,
            ra=source.ra + 180 * U.deg,
            dec=source.dec,
        )
        source._init_skycoords()
        source._init_pixcoords(datacube)
        print(source.pixcoords)
        assert np.isnan(source.pixcoords[0][0])
        assert np.isnan(source.pixcoords[1][0])
        m = Martini(
            source=source,
            datacube=datacube,
            beam=GaussianBeam(),
            noise=None,
            sph_kernel=DiracDeltaKernel(),
            spectral_model=DiracDeltaSpectrum(),
        )
        assert m.source.pixcoords.shape == (3, 1)

    @pytest.mark.parametrize(
        ("prune_kwargs", "expected_npart"),
        (
            # 5 RA x 5 Dec x 5 velocity x 2 mass offsets, of which 3 RA, 3 Dec and 3
            # velocity offsets are close enough to the cube to survive pruning
            ({"spatial": True, "spectral": False, "mass": False}, 3 * 3 * 5 * 2),
            ({"spatial": False, "spectral": True, "mass": False}, 5 * 5 * 3 * 2),
            ({"spatial": False, "spectral": False, "mass": True}, 5 * 5 * 5 * 1),
            ({"spatial": True, "spectral": True, "mass": True}, 3 * 3 * 3 * 1),
        ),
    )
    def test_prune_does_not_change_cube(
        self, prune_kwargs, expected_npart, prune_test_cube
    ):
        """
        Check that pruning particles does not change the resulting datacube.

        Pruning is just done to remove unnecessary work downsteam, it should have no
        influence on the actual result.

        The test source (see ``prune_test_cubegen`` in ``conftest.py``) has particles
        inside, overlapping the edges of, and far outside the cube both on the sky and in
        velocity, each with zero and non-zero HI mass. The kernel has compact support,
        so the cubes should agree with and without pruning. Check that the pruning
        conditions remove exactly the particles that cannot contribute, and that some
        particles survive including some that overlap the edges of the cube.
        """
        n, npart_pruned, cube_pruned = prune_test_cube(prune_kwargs)
        _, npart_unpruned, cube_unpruned = prune_test_cube(
            {"spatial": False, "spectral": False, "mass": False}
        )
        assert npart_unpruned == n
        assert npart_pruned == expected_npart
        # particles overlapping the edges must survive and contribute there
        for edge in (
            cube_pruned[0],
            cube_pruned[-1],
            cube_pruned[:, 0],
            cube_pruned[:, -1],
            cube_pruned[:, :, 0],
            cube_pruned[:, :, -1],
        ):
            assert edge.sum() > 0
        assert U.allclose(cube_pruned, cube_unpruned)

    def test_xy_independence(self):
        """
        Check that the x (RA) and y (Dec) axes are handled independently.

        Create a non-square cube with a non-square beam kernel (so the padding differs
        along x and y) and an asymmetric source. Then rotate the source by 90deg about
        the line of sight, swap the cube and kernel dimensions and rotate the beam
        position angle by 90deg. The TAN projection is symmetric under rotation about
        the tangent point, so the second cube should be the first rotated by 90deg.
        """

        class SizedGaussianBeam(GaussianBeam):
            """A Gaussian beam with a given kernel size."""

            def __init__(self, size_px, **kwargs):
                super().__init__(**kwargs)
                self.size_px = size_px

            def kernel_size_px(self):
                return self.size_px

        def make_cube(rotate):
            # an asymmetric "L" of particles in the plane of the sky (y-z)
            y = np.array([0.0, 0.4, 0.8, 1.2, 0.0, 0.0])
            z = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0])
            n = y.size
            source = SPHSource(
                xyz_g=U.Quantity([np.zeros(n), y, z], U.kpc).T,
                vxyz_g=U.Quantity(
                    [np.linspace(-20, 20, n), np.zeros(n), np.zeros(n)], U.km / U.s
                ).T,
                T_g=np.ones(n) * 1.0e4 * U.K,
                hsm_g=np.ones(n) * 0.2 * U.kpc,
                mHI_g=np.ones(n) * 1.0e6 * U.Msun,
                distance=3 * U.Mpc,
                ra=0 * U.deg,
                dec=0 * U.deg,
            )
            if rotate:
                source.rotate(Rotation.from_euler("x", 90, degrees=True))
            n_px = (24, 16) if rotate else (16, 24)
            datacube = DataCube(
                n_px_x=n_px[0],
                n_px_y=n_px[1],
                n_channels=8,
                px_size=10 * U.arcsec,
                channel_width=8 * U.km / U.s,
                spectral_centre=source.vsys,
                ra=0 * U.deg,
                dec=0 * U.deg,
            )
            beam = SizedGaussianBeam(
                (7, 4) if rotate else (4, 7),
                bmaj=30 * U.arcsec,
                bmin=15 * U.arcsec,
                bpa=(90 if rotate else 0) * U.deg,
            )
            m = Martini(
                source=source,
                datacube=datacube,
                beam=beam,
                sph_kernel=_CubicSplineKernel(),
                spectral_model=GaussianSpectrum(sigma=10 * U.km / U.s),
                quiet=True,
            )
            pad = (m.datacube.padx, m.datacube.pady)
            m.insert_source_in_cube(skip_validation=True)
            m.convolve_beam()
            return pad, m.datacube._array

        pad, cube = make_cube(rotate=False)
        pad_rotated, cube_rotated = make_cube(rotate=True)
        assert pad == (4, 7)
        assert pad_rotated == (7, 4)
        assert cube.shape == (16, 24, 8)
        assert cube.max() > 0
        assert U.allclose(
            np.rot90(cube, k=3, axes=(0, 1)),
            cube_rotated,
            rtol=1e-6,
            atol=1e-9 * cube.max(),
        )

    @pytest.mark.parametrize("bpa", (0 * U.deg, 45 * U.deg, 90 * U.deg, 135 * U.deg))
    def test_beam_position_angle_on_sky(self, bpa):
        """
        Check that the beam convolved into the cube has the expected orientation.

        A point source is convolved with an elliptical beam in a non-square cube, and the
        orientation of the resulting image is measured on the sky using the WCS.
        """
        source = SPHSource(
            xyz_g=np.ones((1, 3)) * 1.0e-3 * U.kpc,  # avoid landing on a pixel edge
            vxyz_g=np.zeros((1, 3)) * U.km / U.s,
            T_g=np.ones(1) * 1.0e4 * U.K,
            hsm_g=np.ones(1) * 1.0e-3 * U.kpc,
            mHI_g=np.ones(1) * 1.0e8 * U.Msun,
            distance=3 * U.Mpc,
            ra=10 * U.deg,
            dec=20 * U.deg,
        )
        datacube = DataCube(
            n_px_x=80,
            n_px_y=60,
            n_channels=5,
            px_size=1 * U.arcsec,
            channel_width=4 * U.km / U.s,
            spectral_centre=source.vsys,
            ra=source.ra,
            dec=source.dec,
        )
        m = Martini(
            source=source,
            datacube=datacube,
            beam=GaussianBeam(
                bmaj=20 * U.arcsec, bmin=5 * U.arcsec, bpa=bpa, truncate=3
            ),
            sph_kernel=DiracDeltaKernel(),
            spectral_model=GaussianSpectrum(sigma=4 * U.km / U.s),
            quiet=True,
        )
        m.insert_source_in_cube(skip_validation=True)
        m.convolve_beam()
        mom0 = np.clip(m.datacube._array.sum(axis=2).value, 0, None)
        i, j = np.meshgrid(
            np.arange(mom0.shape[0]), np.arange(mom0.shape[1]), indexing="ij"
        )
        ra, dec = m.datacube.wcs.celestial.wcs_pix2world(i.ravel(), j.ravel(), 0)
        east, north = SkyCoord(source.ra, source.dec).spherical_offsets_to(
            SkyCoord(ra * U.deg, dec * U.deg)
        )
        # orientation of the major axis from the second moments of the image
        cov = np.cov(
            np.vstack((east.to_value(U.arcsec), north.to_value(U.arcsec))),
            aweights=mom0.ravel(),
        )
        major_axis = np.linalg.eigh(cov)[1][:, -1]
        measured_pa = np.arctan2(major_axis[0], major_axis[1]) * U.rad
        # position angle is only defined modulo 180deg
        offset = (measured_pa - bpa + 90 * U.deg) % (180 * U.deg) - 90 * U.deg
        assert np.abs(offset) < 1 * U.deg

    def test_asymmetric_beam_orientation_on_sky(self):
        """
        Check the orientation on the sky of a beam that is not symmetric.

        A Gaussian beam is symmetric, so a beam flipped along both axes goes undetected.
        Convolve a point source with a beam offset to the North-East of its centre and
        check where the flux lands on the sky using the WCS.
        """
        source = SPHSource(
            xyz_g=np.ones((1, 3)) * 1.0e-3 * U.kpc,  # avoid landing on a pixel edge
            vxyz_g=np.zeros((1, 3)) * U.km / U.s,
            T_g=np.ones(1) * 1.0e4 * U.K,
            hsm_g=np.ones(1) * 1.0e-3 * U.kpc,
            mHI_g=np.ones(1) * 1.0e8 * U.Msun,
            distance=3 * U.Mpc,
            ra=10 * U.deg,
            dec=20 * U.deg,
        )
        datacube = DataCube(
            n_px_x=40,
            n_px_y=30,
            n_channels=5,
            px_size=1 * U.arcsec,
            channel_width=4 * U.km / U.s,
            spectral_centre=source.vsys,
            ra=source.ra,
            dec=source.dec,
        )
        m = Martini(
            source=source,
            datacube=datacube,
            beam=OffsetBeam(bmaj=3 * U.arcsec),
            sph_kernel=DiracDeltaKernel(),
            spectral_model=GaussianSpectrum(sigma=4 * U.km / U.s),
            quiet=True,
        )
        m.insert_source_in_cube(skip_validation=True)
        m.convolve_beam()
        mom0 = np.clip(m.datacube._array.sum(axis=2).value, 0, None)
        i, j = np.meshgrid(
            np.arange(mom0.shape[0]), np.arange(mom0.shape[1]), indexing="ij"
        )
        ra, dec = m.datacube.wcs.celestial.wcs_pix2world(
            np.average(i, weights=mom0), np.average(j, weights=mom0), 0
        )
        east, north = SkyCoord(source.ra, source.dec).spherical_offsets_to(
            SkyCoord(ra * U.deg, dec * U.deg)
        )
        # the point source is deposited in the nearest pixel, allow for that offset
        assert np.abs(east - OffsetBeam.offset_east) < 1 * U.arcsec
        assert np.abs(north - OffsetBeam.offset_north) < 1 * U.arcsec

    def test_reset(self, m_nn):
        """Check that resetting martini instance zeros out datacube."""
        cube_array = m_nn.datacube._array
        assert m_nn.datacube._array.sum() > 0
        m_nn.reset()
        assert m_nn.datacube._cube_data is None
        # check that can start over and get the same result w/o errors
        m_nn.insert_source_in_cube(progressbar=False)
        m_nn.convolve_beam()
        assert U.allclose(cube_array, m_nn.datacube._array)
        # check that can reset after doing nothing
        m_nn.reset()
        m_nn.reset()

    def test_reset_preserves_shape(self, single_particle_source, dc_zeros):
        """Check that using the reset feature preserves the shape of the datacube."""
        m = Martini(
            source=single_particle_source(),
            datacube=dc_zeros,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=DiracDeltaSpectrum(),
            sph_kernel=DiracDeltaKernel(),
        )
        expected_shape = m.datacube._array.shape
        m.reset()
        m._datacube._allocate_cube()
        assert m.datacube._array.shape == expected_shape

    def test_preview(self, m_init, tmp_path):
        """Simply check that the preview visualisation runs without error."""
        pytest.importorskip(
            "matplotlib", reason="matplotlib (optional dependency) not available."
        )
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        # with default arguments
        plt.close(m_init.preview())
        # with non-default arguments
        plt.close(
            m_init.preview(
                max_points=1000,
                lim="datacube",
                vlim="datacube",
                point_scaling="fixed",
                title="test",
            )
        )
        # saving to file
        plt.close(m_init.preview(save=tmp_path / "preview.png"))
        assert (tmp_path / "preview.png").exists()

    def test_preview_warning_ra_dec(self, m_init):
        """Check that we warn if the datacube is too big in RA."""
        pytest.importorskip(
            "matplotlib", reason="matplotlib (optional dependency) not available."
        )
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        m_init._datacube.px_size = 20 * U.deg / m_init._datacube.n_px_x
        assert m_init._datacube.px_size * m_init._datacube.n_px_x > 10 * U.deg
        assert m_init._datacube.px_size * m_init._datacube.n_px_y > 10 * U.deg
        with (
            pytest.warns(UserWarning, match="DataCube RA range > 5deg"),
            pytest.warns(UserWarning, match="DataCube Dec range > 5deg"),
        ):
            plt.close(m_init.preview())

    @pytest.mark.parametrize("dec", (89 * U.deg, -89 * U.deg))
    def test_preview_warning_pole(self, m_init, dec):
        """Check that we warn if the datacube is too close to the pole."""
        pytest.importorskip(
            "matplotlib", reason="matplotlib (optional dependency) not available."
        )
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        m_init._datacube.dec = dec
        assert np.abs(m_init._datacube.dec) - 90 * U.deg < 5 * U.deg
        with (
            pytest.warns(UserWarning, match="DataCube extent within 5deg of pole"),
        ):
            plt.close(m_init.preview())

    def test_source_to_datacube_coord_transformation(self, single_particle_source):
        """Check transformation is applied if source and datacube frames differ."""
        source = single_particle_source(hsm_g=0.01 * U.kpc)
        assert source.coordinate_frame.name == "icrs"
        datacube_icrs = DataCube(
            n_px_x=16,
            n_px_y=16,
            n_channels=16,
            channel_width=4 * U.km / U.s,
            px_size=10 * U.arcsec,
            spectral_centre=source.vsys,
            ra=source.ra,
            dec=source.dec,
            coordinate_frame=ICRS(),
        )
        m_icrs = Martini(
            source=source,
            datacube=datacube_icrs,
            beam=GaussianBeam(),
            noise=None,
            sph_kernel=DiracDeltaKernel(),
            spectral_model=DiracDeltaSpectrum(),
        )

        def centre_pixels_slice(m):
            """
            Get 1/4 of the cube (excluding pad region) centered in RA and Dec.

            Parameters
            ----------
            m : ~martini.martini.Martini
                The martini object to get the region from.

            Returns
            -------
            np.ndarray
                The sub-region of the cube.
            """
            datacube = m.datacube
            return m.datacube._array[
                datacube.n_px_x // 2 - 1 + datacube.padx : datacube.n_px_x // 2
                + 1
                + datacube.padx,
                datacube.n_px_y // 2 - 1 + datacube.pady : datacube.n_px_y // 2
                + 1
                + datacube.pady,
            ]

        m_icrs._datacube._allocate_cube()
        assert np.sum(centre_pixels_slice(m_icrs).sum()) == 0
        m_icrs.insert_source_in_cube(progressbar=False)
        assert np.sum(centre_pixels_slice(m_icrs).sum()) > 0

        # ICRS is ~J2000 equinox. J1950 equinox is about a degree off,
        # so we should completely miss the cube (16 pix of 10 arcsec).
        datacube_fk5_J1950 = DataCube(
            n_px_x=16,
            n_px_y=16,
            n_channels=16,
            channel_width=4 * U.km / U.s,
            px_size=10 * U.arcsec,
            spectral_centre=source.vsys,
            ra=source.ra,
            dec=source.dec,
            coordinate_frame=FK5(equinox="J1950"),
        )
        with pytest.raises(
            RuntimeError,
            match="No non-zero mHI source particles in target region.",
        ):
            Martini(
                source=source,
                datacube=datacube_fk5_J1950,
                beam=GaussianBeam(),
                noise=None,
                sph_kernel=DiracDeltaKernel(),
                spectral_model=DiracDeltaSpectrum(),
            )

    def test_source_to_datacube_specsys_transformation(self, single_particle_source):
        """
        Check that spectral reference transformation is applied when needed.

        If the source and datacube have different specsys, we need to apply a coordinate
        transformation when inserting the particles. Make sure that this happens.
        """
        source = single_particle_source(hsm_g=0.01 * U.kpc)
        datacube_icrs = DataCube(
            n_px_x=16,
            n_px_y=16,
            n_channels=16,
            channel_width=4 * U.km / U.s,
            px_size=10 * U.arcsec,
            spectral_centre=source.vsys,
            ra=source.ra,
            dec=source.dec,
            coordinate_frame=ICRS(),
            specsys="icrs",
        )
        m_icrs = Martini(
            source=source,
            datacube=datacube_icrs,
            beam=GaussianBeam(),
            noise=None,
            sph_kernel=DiracDeltaKernel(),
            spectral_model=DiracDeltaSpectrum(),
        )

        def centre_channels_slice(m):
            """
            Get the central half of the spectral channels from a cube.

            Parameters
            ----------
            m : ~martini.martini.Martini
                The martini object to get the region from.

            Returns
            -------
            np.ndarray
                The sub-region of the cube.
            """
            datacube = m.datacube
            return m.datacube._array[
                :, :, datacube.n_channels // 2 - 1 : datacube.n_channels // 2 + 1
            ]

        m_icrs._datacube._allocate_cube()
        assert np.sum(centre_channels_slice(m_icrs).sum()) == 0
        m_icrs.insert_source_in_cube(progressbar=False)
        assert np.sum(centre_channels_slice(m_icrs).sum()) > 0

        # ICRS and Galactocentric are offset by many km/s depending on direction
        # so with 4 channels of 1 km/s we should completely miss the cube
        datacube_galactocentric = DataCube(
            n_px_x=16,
            n_px_y=16,
            n_channels=4,
            channel_width=1 * U.km / U.s,
            px_size=10 * U.arcsec,
            spectral_centre=source.vsys,
            ra=source.ra,
            dec=source.dec,
            coordinate_frame=ICRS(),
            specsys="galactocentric",
        )
        assert datacube_galactocentric.wcs.wcs.specsys == "galactocentric"
        with pytest.raises(
            RuntimeError,
            match="No non-zero mHI source particles in target region.",
        ):
            Martini(
                source=source,
                datacube=datacube_galactocentric,
                beam=GaussianBeam(),
                noise=None,
                sph_kernel=DiracDeltaKernel(),
                spectral_model=DiracDeltaSpectrum(),
            )

    def test_hdf5_grids(self, m, tmp_path_factory):
        """Check the explicit RA, Dec and spectral grid optionally output in HDF5 mode."""
        h5py = pytest.importorskip(
            "h5py", reason="h5py (optional dependency) not available."
        )
        origin = 0  # numpy-like, not fits-like, indexing
        filename = str(tmp_path_factory.mktemp("cube") / "cube.hdf5")
        m.write_hdf5(filename)
        with h5py.File(filename, "r") as f:
            voxel_coords = np.vstack(
                (
                    (
                        f["RA"][()].flatten()
                        * U.Unit(f["RA"].attrs["Unit"], format="fits")
                    ).to_value(m.datacube.wcs.wcs.cunit[0]),
                    (
                        f["Dec"][()].flatten()
                        * U.Unit(f["Dec"].attrs["Unit"], format="fits")
                    ).to_value(m.datacube.wcs.wcs.cunit[1]),
                    (
                        f["channel_mids"][()].flatten()
                        * U.Unit(f["channel_mids"].attrs["Unit"], format="fits")
                    ).to_value(m.datacube.wcs.wcs.cunit[2]),
                )
            ).T
            ra_idx, dec_idx, spec_idx = m.datacube.wcs.all_world2pix(
                voxel_coords, origin
            ).T
            ra_idx = ra_idx.reshape((m.datacube._array.shape))
            dec_idx = dec_idx.reshape((m.datacube._array.shape))
            spec_idx = spec_idx.reshape((m.datacube._array.shape))
            expected_idx = np.meshgrid(
                np.arange(m.datacube.n_px_x),
                np.arange(m.datacube.n_px_y),
                np.arange(m.datacube.n_channels),
                indexing="ij",
            )
            assert np.allclose(ra_idx, expected_idx[0])
            assert np.allclose(dec_idx, expected_idx[1])
            assert np.allclose(spec_idx, expected_idx[2])
            vertex_coords = np.vstack(
                (
                    (
                        f["RA_vertices"][()].flatten()
                        * U.Unit(f["RA_vertices"].attrs["Unit"], format="fits")
                    ).to_value(m.datacube.wcs.wcs.cunit[0]),
                    (
                        f["Dec_vertices"][()].flatten()
                        * U.Unit(f["Dec_vertices"].attrs["Unit"], format="fits")
                    ).to_value(m.datacube.wcs.wcs.cunit[1]),
                    (
                        f["channel_vertices"][()].flatten()
                        * U.Unit(f["channel_vertices"].attrs["Unit"], format="fits")
                    ).to_value(m.datacube.wcs.wcs.cunit[2]),
                )
            ).T
            ra_vx_idx, dec_vx_idx, spec_vx_idx = m.datacube.wcs.all_world2pix(
                vertex_coords, origin
            ).T
            shape = [s + 1 for s in m.datacube._array.shape]
            ra_vx_idx = ra_vx_idx.reshape(shape)
            dec_vx_idx = dec_vx_idx.reshape(shape)
            spec_vx_idx = spec_vx_idx.reshape(shape)
            expected_vx_idx = np.meshgrid(
                np.arange(m.datacube.n_px_x + 1) - 0.5,
                np.arange(m.datacube.n_px_y + 1) - 0.5,
                np.arange(m.datacube.n_channels + 1) - 0.5,
                indexing="ij",
            )
            assert np.allclose(ra_vx_idx, expected_vx_idx[0])
            assert np.allclose(dec_vx_idx, expected_vx_idx[1])
            assert np.allclose(spec_vx_idx, expected_vx_idx[2])


class TestParallel:
    """Test that results running in parallel agree with those running serially."""

    def test_parallel_consistent_with_serial(self, many_particle_source, dc_zeros):
        """
        Check running the source insertion loop in parallel.

        Should give the same result as running in serial. The serial "fallback"
        implementation is the reference case, the numba implementation is checked
        against it.
        """
        pytest.importorskip("numba", reason="numba (optional dependency) not available")

        m = Martini(
            source=many_particle_source(),
            datacube=dc_zeros,
            beam=GaussianBeam(),
            noise=None,
            sph_kernel=_GaussianKernel(),
            spectral_model=GaussianSpectrum(),
        )

        m._numba_enabled = False
        m.insert_source_in_cube(ncpu=1, progressbar=False)  # fallback implementation

        expected_result = m.datacube._array

        # check that we're not testing on a zero array
        assert m.datacube._array.sum() > 0

        m.reset()

        # check the reset was successful
        assert m.datacube._cube_data is None

        m._numba_enabled = True
        m.insert_source_in_cube(progressbar=False)

        assert U.allclose(m.datacube._array, expected_result)

    @pytest.mark.parametrize("sph_kernel", simple_kernels + recommended_kernels)
    def test_weight_and_insert_matches_fallback(
        self, many_particle_source, dc_zeros, sph_kernel
    ):
        """Check that evaluating kernel weights on the fly matches precomputed weights."""
        pytest.importorskip("numba", reason="numba (optional dependency) not available")
        m = Martini(
            source=many_particle_source(),
            datacube=dc_zeros,
            beam=GaussianBeam(),
            noise=None,
            sph_kernel=sph_kernel(),
            spectral_model=GaussianSpectrum(),
        )
        assert m.sph_kernel._numba_kernel_arrays() is not None
        m._numba_enabled = False
        m.insert_source_in_cube(skip_validation=True, progressbar=False)
        expected_result = m.datacube._array.copy()
        assert expected_result.sum() > 0
        m.reset()
        m._numba_enabled = True
        m.insert_source_in_cube(skip_validation=True, progressbar=False)
        assert U.allclose(m.datacube._array, expected_result)

    def test_kernel_without_compiled_integral(
        self, many_particle_source, dc_zeros, monkeypatch
    ):
        """Check that a kernel without a compiled integral is an error with numba."""
        m = Martini(
            source=many_particle_source(),
            datacube=dc_zeros,
            beam=GaussianBeam(),
            noise=None,
            sph_kernel=_GaussianKernel(),
            spectral_model=GaussianSpectrum(),
        )
        m.insert_source_in_cube(progressbar=False)
        expected_result = m.datacube._array.copy()
        m.reset()
        monkeypatch.setattr(
            m.sph_kernel, "_scalar_kernel_integral", None, raising=False
        )
        with pytest.raises(ValueError, match="_scalar_kernel_integral"):
            m.insert_source_in_cube(progressbar=False)
        # without numba the (uncompiled) fallback still works:
        m.reset()
        m._numba_enabled = False
        m.insert_source_in_cube(progressbar=False)
        assert U.allclose(m.datacube._array, expected_result)

    def test_insert_without_overlaps(self, m_init):
        """Check that a batch of particles outside of the cube leaves it unchanged."""
        kernel_arrays = m_init.sph_kernel._numba_kernel_arrays()
        assert kernel_arrays is not None
        gs = FindGridIntersectionsResult(
            intersections=np.empty(0, dtype=np.int32),
            distances=None,
            cell_indices=np.empty(0, dtype=np.int64),
            strides=np.empty((0, 2), dtype=np.int64),
        )
        before = m_init.datacube._array.copy()
        m_init._insert_weighted_spectra(
            np.empty((0, m_init.datacube.n_channels)), gs, slice(0, 0), kernel_arrays, 1
        )
        assert U.allclose(m_init.datacube._array, before)

    def test_numba_related_warnings(self, m_init):
        """Check that we warn the user that ncpu>1 doesn't work without numba."""
        m_init._numba_enabled = False
        with pytest.warns(RuntimeWarning, match="Parallelization not available"):
            m_init.insert_source_in_cube(ncpu=2)
        m_init.reset()
        m_init._numba_enabled = True
        with pytest.warns(
            RuntimeWarning,
            match="'numba'-accelerated 'martini' does not support progress bar",
        ):
            m_init.insert_source_in_cube(progressbar=True)

    def test_all_cores(self, m_init):
        """Check that ncpu=-1 runs and agrees with a single thread."""
        m_init.insert_source_in_cube(ncpu=1, progressbar=False)
        expected_result = m_init.datacube._array.copy()
        assert expected_result.sum() > 0
        m_init.reset()
        m_init.insert_source_in_cube(ncpu=-1, progressbar=False)
        assert U.allclose(m_init.datacube._array, expected_result)

    def test_resolve_ncpu(self, monkeypatch):
        """Check conversion of requested thread counts."""
        assert resolve_ncpu(3) == 3
        assert resolve_ncpu(-1) >= 1
        monkeypatch.setattr(martini._util, "NUMBA_AVAILABLE", False)
        assert resolve_ncpu(-1) == 1
        for bad in (0, -2):
            with pytest.raises(ValueError, match="ncpu must be"):
                resolve_ncpu(bad)


class TestBatched:
    """Test that results running in particle batches agree with running all at once."""

    @pytest.mark.parametrize("numba_on", (True, False))
    def test_batched_consistent_with_concurrent(
        self, many_particle_source, dc_zeros, numba_on
    ):
        """
        Check running the source insertion loop in batches.

        Should give the same result as running in one batch.
        """
        m = Martini(
            source=many_particle_source(),
            datacube=dc_zeros,
            beam=GaussianBeam(),
            noise=None,
            sph_kernel=_GaussianKernel(),
            spectral_model=GaussianSpectrum(),
        )

        m._numba_enabled = numba_on
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", category=RuntimeWarning, message="Requested memory limit"
            )
            m._insert_source_in_cube(
                progressbar=False, mem_lim_GB=m.baseline_mem_estimate + 0.1
            )

        expected_result = m.datacube._array

        # check that we're not testing on a zero array
        assert m.datacube._array.sum() > 0

        m.reset()

        # check the reset was successful
        assert m.datacube._cube_data is None

        # set the memory limit to allow about half of the particles per batch:
        sm_ranges = np.clip(
            m.sph_kernel.sm_ranges.to_value(U.pix), np.sqrt(2) / 2, np.inf
        )
        weight_and_insert = numba_on
        mem_estimate, overhead = m._insertion_mem_estimate(
            m._count_px_hits(sm_ranges, 1), sm_ranges, weight_and_insert
        )
        assert mem_estimate.max() < 0.5 * mem_estimate.sum()
        with pytest.warns(RuntimeWarning, match="Requested memory limit"):
            m._insert_source_in_cube(
                progressbar=False,
                mem_lim_GB=m.baseline_mem_estimate
                + overhead
                + 0.5 * mem_estimate.sum(),
            )

        assert U.allclose(m.datacube._array, expected_result)

    def test_plan_segments(self):
        """Check that particle batches are contiguous and respect the allowance."""
        rng = np.random.default_rng(0)
        mem_estimate = rng.random(100)
        allowance = 3.0
        # make sure that batching is actually needed:
        assert mem_estimate.sum() > allowance
        segments = _plan_segments(mem_estimate, allowance)
        assert len(segments) > 1
        assert segments[0].start == 0
        assert segments[-1].stop == mem_estimate.size
        for s0, s1 in zip(segments[:-1], segments[1:]):
            assert s0.stop == s1.start
        for segment in segments:
            assert segment.stop > segment.start
            assert mem_estimate[segment].sum() <= allowance
        with pytest.raises(ValueError, match="insufficient for a single particle"):
            _plan_segments(np.array([1.0, 4.0, 1.0]), allowance)

    def test_insufficient_memory(self, m_init):
        """Test that insufficient memory to process even one particle crashes."""
        with pytest.warns(RuntimeWarning, match="Requested memory limit"):
            with pytest.raises(RuntimeError, match="Requested memory limit"):
                m_init.insert_source_in_cube(
                    progressbar=False, mem_lim_GB=m_init.baseline_mem_estimate
                )
        with pytest.raises(RuntimeError, match="Requested memory limit"):
            m_init.insert_source_in_cube(
                progressbar=False, mem_lim_GB=m_init.baseline_mem_estimate - 0.0001
            )


class TestGlobalProfile:
    """Test the functionality of the simplified martini module for global profiles."""

    def test_search_bypass_matches_search(self, many_particle_source):
        """Check that GlobalProfile's shortcut gives the same overlaps as a search."""
        source = many_particle_source()
        m = GlobalProfile(
            source=source,
            spectral_model=GaussianSpectrum(),
            n_channels=32,
            channel_width=10 * U.km * U.s**-1,
            spectral_centre=source.vsys,
        )
        sm_ranges = m.sph_kernel.sm_ranges.to_value(U.pix)
        px_hits = m._count_px_hits(sm_ranges, 1)
        assert np.array_equal(px_hits, _BaseMartini._count_px_hits(m, sm_ranges, 1))
        for segment in (slice(None), slice(3, 10), slice(5, 5)):
            gs = m._find_px_hits(segment, sm_ranges, 1)
            gs_search = _BaseMartini._find_px_hits(m, segment, sm_ranges, 1)
            for a, b in zip(gs, gs_search):
                assert np.array_equal(a, b)

    @pytest.mark.parametrize("spectral_model", (DiracDeltaSpectrum, GaussianSpectrum))
    @pytest.mark.parametrize("ra", (0 * U.deg, 180 * U.deg))
    @pytest.mark.parametrize("combined_source", (True, False))
    def test_mass_accuracy(
        self, spectral_model, single_particle_source, ra, combined_source
    ):
        """
        Check that mass in final spectrum agrees with input mass within 1%.

        Check that the input mass in the particles gives the correct total mass in the
        spectrum, by checking the conversion back to total mass. Covers testing
        GlobalProfile.insert_source_in_spectrum.
        """
        # single_particle_source has a mass of 1E4Msun, temperature of 1E4K
        # we test both ra=0deg and ra=180deg to make sure all particles included
        source = single_particle_source(ra=ra)
        if combined_source:
            source = CombinedSource([source, source])
        m = GlobalProfile(
            source=source,
            spectral_model=spectral_model(),
            n_channels=32,
            channel_width=10 * U.km * U.s**-1,
            spectral_centre=source.vsys,
        )

        m.insert_source_in_spectrum()

        # flux
        F = m.spectrum.sum()  # Jy

        # distance
        D = m.source.distance

        # channel width
        dv = m.channel_width

        # HI mass
        MHI = (
            2.36e5
            * U.Msun
            * D.to_value(U.Mpc) ** 2
            * F.to_value(U.Jy)
            * dv.to_value(U.km / U.s)
        ).to(U.Msun)

        # demand accuracy within 1% after source insertion
        assert U.isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

    @pytest.mark.parametrize(
        ("ra_off", "ra_in"),
        (
            (0 * U.arcsec, True),
            (3 * U.arcsec, True),
            (5 * U.deg, False),  # global profile uses 1 deg pixel
            (-3 * U.arcsec, True),
            (-5 * U.deg, False),  # global profile uses 1 deg pixel
        ),
    )
    @pytest.mark.parametrize(
        ("dec_off", "dec_in"),
        (
            (0 * U.arcsec, True),
            (3 * U.arcsec, True),
            (5 * U.deg, False),  # global profile uses 1 deg pixel
            (-3 * U.arcsec, True),
            (-5 * U.deg, False),  # global profile uses 1 deg pixel
        ),
    )
    @pytest.mark.parametrize(
        ("v_off", "v_in"),
        (
            (0 * U.km / U.s, True),
            (3 * U.km / U.s, True),
            (7 * U.km / U.s, False),
            (-3 * U.km / U.s, True),
            (-7 * U.km / U.s, False),
        ),
    )
    def test_prune_particles(
        self, ra_off, ra_in, dec_off, dec_in, v_off, v_in, single_particle_source
    ):
        """
        Test pruning particles unused in the creation of the mock.

        Check that a particle offset by a specific set of (RA, Dec, v) is inside/outside
        the cube as expected.
        """
        # GlobalProfile should ignore RA, Dec when pruning:
        expect_particle = v_in
        # set distance so that 1kpc = 1arcsec
        distance = (1 * U.kpc / 1 / U.arcsec).to(U.Mpc, U.dimensionless_angles())
        source = single_particle_source(
            distance=distance, ra=ra_off, dec=dec_off, vpeculiar=v_off
        )
        spectral_model = GaussianSpectrum(sigma=1 * U.km / U.s)
        kwargs = {
            "source": source,
            "spectral_model": spectral_model,
            "n_channels": 2,
            "channel_width": 1 * U.km / U.s,
            "spectral_centre": source.distance * source.h * 100 * U.km / U.s / U.Mpc,
        }
        # if more than 1px (datacube) + 4px (4*spectrum_half_width) then expect to prune
        if not expect_particle:
            with pytest.raises(
                RuntimeError,
                match="No non-zero mHI source particles in target region.",
            ):
                GlobalProfile(**kwargs)
        else:
            assert GlobalProfile(**kwargs).source.npart == 1

    def test_reset(self, gp):
        """Check that resetting global profile zeros out datacube and spectrum."""
        cube_array = gp._datacube._array
        assert gp._datacube._array.sum() > 0
        spectrum = gp.spectrum
        assert spectrum.sum() > 0
        gp.reset()
        assert gp._datacube._cube_data is None
        assert not hasattr(gp, "_spectrum")
        # check that can start over and get the same result w/o errors
        gp.insert_source_in_spectrum()
        assert U.allclose(cube_array, gp._datacube._array)
        assert U.allclose(spectrum, gp.spectrum)
        # check that can reset after doing nothing
        gp.reset()
        gp.reset()

    def test_preview(self, gp):
        """Simply check that the preview visualisation runs without error."""
        pytest.importorskip(
            "matplotlib", reason="matplotlib (optional dependency) not available."
        )
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        # with default arguments
        plt.close(gp.preview())
        # with non-default arguments
        plt.close(
            gp.preview(
                max_points=1000,
                lim="datacube",
                vlim="datacube",
                point_scaling="fixed",
                title="test",
            )
        )

    def test_channel_modes(self, single_particle_source):
        """Check that channels have expected units in both modes (frequency, velocity)."""
        source = single_particle_source()
        channel_width = 10 * U.km * U.s**-1
        m = GlobalProfile(
            source=source,
            spectral_model=GaussianSpectrum(sigma="thermal"),
            n_channels=32,
            channel_width=channel_width,
            spectral_centre=source.vsys,
        )
        expected_units = channel_width.unit
        # these will raise if there's a problem:
        m.channel_edges.to(expected_units)
        m.channel_mids.to(expected_units)

    def test_frequency_and_velocity_channels(self, gp):
        """Check that channels are available in both frequency and velocity units."""
        for attr in (
            "frequency_channel_edges",
            "frequency_channel_mids",
            "velocity_channel_edges",
            "velocity_channel_mids",
        ):
            assert U.allclose(getattr(gp, attr), getattr(gp._datacube, attr))
        assert gp.frequency_channel_edges.unit.is_equivalent(U.Hz)
        assert gp.velocity_channel_edges.unit.is_equivalent(U.km / U.s)
        assert gp.frequency_channel_edges.size == gp.frequency_channel_mids.size + 1
        assert U.allclose(
            gp.velocity_channel_mids.to(U.Hz, equivalencies=U.doppler_radio(HIfreq)),
            gp.frequency_channel_mids,
        )

    def test_spectrum_inserts_source(self, single_particle_source):
        """Check that accessing the spectrum inserts the source if not yet done."""
        source = single_particle_source()
        m = GlobalProfile(
            source=source,
            spectral_model=GaussianSpectrum(),
            n_channels=16,
            channel_width=4 * U.km / U.s,
            spectral_centre=source.vsys,
            quiet=True,
        )
        assert not hasattr(m, "_spectrum")
        assert m.spectrum.sum() > 0
        assert hasattr(m, "_spectrum")

    def test_view_spectrum(self, gp, tmp_path):
        """Simply check that plotting spectrum runs without error."""
        pytest.importorskip(
            "matplotlib", reason="matplotlib (optional dependency) not available."
        )
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        # with default arguments
        plt.close(gp.plot_spectrum())
        # with non-default arguments
        plt.close(gp.plot_spectrum(title="test", show_vsys=False))
        # in frequency units
        plt.close(gp.plot_spectrum(channels="frequency"))
        # saving to file
        plt.close(gp.plot_spectrum(save=tmp_path / "spectrum.png"))
        assert (tmp_path / "spectrum.png").exists()
        with pytest.raises(ValueError, match="must be 'velocity' or 'frequency'"):
            gp.plot_spectrum(channels="wavelength")


class TestMartiniWithDataCubeFromWCS:
    """Test the feature allowing setting up the datacube from an existing WCS."""

    @pytest.mark.parametrize("out_mode", ("fits", "hdf5"))
    def test_source_insertion(
        self, dc_wcs, single_particle_source, out_mode, tmp_path_factory
    ):
        """Check that the mass in the final cube agrees with input mass within 1%."""
        datacube = dc_wcs
        distance = (
            datacube.spectral_centre.to(
                U.km / U.s, equivalencies=U.doppler_radio(HIfreq)
            )
            / (70 * U.km / U.s / U.Mpc)
        ).to(U.Mpc)
        source = single_particle_source(
            ra=datacube.ra,
            dec=datacube.dec,
            distance=distance,
            hsm_g=(3 * datacube.px_size * distance).to(
                U.kpc, equivalencies=U.dimensionless_angles()
            ),
        )
        beam = GaussianBeam(bmaj=3 * datacube.px_size, bmin=3 * datacube.px_size)
        m = Martini(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=None,
            spectral_model=GaussianSpectrum(sigma="thermal"),
            sph_kernel=_CubicSplineKernel(),
        )
        m.insert_source_in_cube(progressbar=False)
        check_mass_accuracy(m, out_mode, tmp_path_factory.mktemp("cube"))


class TestLogging:
    """Test the progress messages logged by Martini."""

    def test_step_messages(self, m_init, capsys, tmp_path):
        """Check start and end messages of each main step, and their formatting."""
        # pruning (at initialization) was logged:
        assert any("Pruning particles" in record for record in m_init._logger.records)
        m_init.insert_source_in_cube(progressbar=False)
        m_init.add_noise()
        m_init.convolve_beam()
        m_init.write_fits(tmp_path / "cube.fits")
        lines = capsys.readouterr().out.splitlines()
        assert all(re.match(r"^\[\d\d:\d\d:\d\d\] ", line) for line in lines)
        messages = [line[11:] for line in lines]
        for start, end in (
            (
                "Inserting source in data cube...",
                "Source inserted in data cube, took ",
            ),
            ("Adding noise...", "Noise added, took "),
            ("Convolving beam...", "Beam convolved, took "),
            ("Writing data cube to FITS file...", "Wrote FITS file "),
        ):
            assert start in messages
            assert any(message.startswith(end) for message in messages)
        # stages of source insertion are indented under it:
        assert "  Counting particle-pixel overlaps..." in messages
        assert any(
            message.startswith("  Reduced pixel spectra into cube, took ")
            for message in messages
        )
        # summary details are indented below the end message:
        assert any(
            message.startswith("  Flux density in cube: ") for message in messages
        )

    def test_quiet(self, m_init, capsys):
        """Check that a quiet run prints nothing but still records its log."""
        m_init.quiet = True
        m_init.insert_source_in_cube(progressbar=False)
        assert capsys.readouterr().out == ""
        assert any(
            record[11:].startswith("Source inserted in data cube, took ")
            for record in m_init._logger.records
        )

    def test_global_profile_messages(self, single_particle_source):
        """Check that GlobalProfile names the spectrum in its messages."""
        source = single_particle_source()
        m = GlobalProfile(
            source=source,
            spectral_model=GaussianSpectrum(),
            n_channels=32,
            channel_width=10 * U.km * U.s**-1,
            spectral_centre=source.vsys,
            quiet=True,
        )
        m.insert_source_in_spectrum()
        messages = [record[11:] for record in m._logger.records]
        assert "Pruning particles that will not contribute to spectrum..." in messages
        assert "Inserting source in spectrum..." in messages
        assert any(
            message.startswith("Source inserted in spectrum, took ")
            for message in messages
        )
