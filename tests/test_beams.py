"""Test the telescope beam modules."""

import pytest
import numpy as np
from astropy import units as U
from martini.beams import GaussianBeam
from martini import DataCube


class TestGaussianBeam:
    """Test the Gaussian beam module."""

    def test_Gaussian_beam_size(self):
        """Check that we produce a square beam array large enough to contain it."""
        px_size = 1 * U.arcsec
        bmaj = 20 * U.arcsec
        bmin = 10 * U.arcsec
        bpa = 0 * U.deg
        truncate = 4
        d = DataCube(
            n_px_x=1,
            n_px_y=1,
            n_channels=1,
            px_size=px_size,
            channel_width=1 * U.km / U.s,
        )
        b = GaussianBeam(bmaj=bmaj, bmin=bmin, bpa=bpa, truncate=truncate)
        b.init_kernel(d)
        halfsize = (
            int(np.ceil((bmaj * truncate / px_size).to_value(U.dimensionless_unscaled)))
            + 1
        )
        expected_size = 2 * halfsize + 1
        assert b.kernel.shape == (expected_size, expected_size)

    @pytest.mark.parametrize("bpa", (0 * U.deg, 90 * U.deg))
    def test_Gaussian_beam_rotation(self, bpa):
        """Check that the beam amplitude is equal along the beam's elliptical outline."""
        px_size = 1 * U.arcsec
        bmaj = 20 * U.arcsec
        bmin = 10 * U.arcsec
        truncate = 4
        d = DataCube(
            n_px_x=1,
            n_px_y=1,
            n_channels=1,
            px_size=px_size,
            channel_width=1 * U.km / U.s,
        )
        b = GaussianBeam(bmaj=bmaj, bmin=bmin, bpa=bpa, truncate=truncate)
        b.init_kernel(d)
        mid = b.kernel.shape[0] // 2
        if bpa == 0 * U.deg:
            assert U.isclose(
                b.kernel[mid, mid + 5],
                b.kernel[mid + 10, mid],
                rtol=1e-2,
            )
        elif bpa == 90 * U.deg:
            assert U.isclose(
                b.kernel[mid, mid + 10],
                b.kernel[mid + 5, mid],
                rtol=1e-2,
            )

    def test_Gaussian_beam_integral(self):
        """Check that the beam integrates to 1.0."""
        px_size = 1 * U.arcsec
        bmaj = 10 * U.arcsec
        bmin = 10 * U.arcsec
        truncate = 4
        d = DataCube(
            n_px_x=1,
            n_px_y=1,
            n_channels=1,
            px_size=px_size,
            channel_width=1 * U.km / U.s,
        )
        b = GaussianBeam(bmaj=bmaj, bmin=bmin, bpa=0 * U.deg, truncate=truncate)
        b.init_kernel(d)
        assert U.isclose(np.sum(b.kernel), 1.0 * U.dimensionless_unscaled)

    def test_Gaussian_bmin_implicit(self):
        """Check that not providing bmin defaults to circular beam."""
        bmaj = 10 * U.arcsec
        bmin = None
        b = GaussianBeam(bmaj=bmaj, bmin=bmin)
        assert b.bmin == b.bmaj


class TestBaseBeam:
    """Test generic beam functionality."""

    def test_needs_pad_uninitialized(self):
        """Check that requesting the padding before initializing the kernel raises."""
        with pytest.raises(RuntimeError, match="Beam kernel not initialized."):
            GaussianBeam().needs_pad()

    def test_deferred_beam_header(self):
        """Check that the beam header is initialized with the kernel if not set."""

        class DeferredGaussianBeam(GaussianBeam):
            """A beam with header initialized only when the kernel is initialized."""

            def __init__(self) -> None:
                super().__init__()
                self.bmaj = self.bmin = self.bpa = None

            def init_beam_header(self) -> None:
                self.bmaj = self.bmin = 15 * U.arcsec
                self.bpa = 0 * U.deg

        d = DataCube(
            n_px_x=1,
            n_px_y=1,
            n_channels=1,
            px_size=5 * U.arcsec,
            channel_width=1 * U.km / U.s,
        )
        b = DeferredGaussianBeam()
        b.init_kernel(d)
        reference = GaussianBeam(bmaj=15 * U.arcsec)
        reference.init_kernel(d)
        assert b.bmaj == reference.bmaj
        assert U.allclose(b.kernel, reference.kernel)
