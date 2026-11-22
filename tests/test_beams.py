"""Test the telescope beam modules."""

import pytest
import numpy as np
from astropy import units as U
from martini.beams import GaussianBeam, _BaseBeam
from martini import DataCube


class OffsetBeam(_BaseBeam):
    """
    A beam that is not symmetric, for testing beam orientation.

    The beam is a circular Gaussian offset 6 arcsec East and 3 arcsec North of the
    beam centre, so that any reflection or transposition of the beam is detectable.
    """

    offset_east = 6 * U.arcsec
    offset_north = 3 * U.arcsec

    def f_kernel(self):
        """
        Return a function defining the beam amplitude as a function of position.

        Returns
        -------
        callable
            Accepts 2 arguments (RA and Dec offsets) and returns the beam amplitude.
        """
        sigma = 1.5 * U.arcsec
        return lambda x, y: (
            np.exp(
                -((x - self.offset_east) ** 2 + (y - self.offset_north) ** 2)
                / (2 * sigma**2)
            )
            / (2 * np.pi * sigma**2)
        )

    def kernel_size_px(self):
        """
        Return a 2-tuple specifying the half-size of the beam image to be initialized.

        Returns
        -------
        tuple
            2-tuple, each element an integer, specifying the kernel size (x, y) in pixels.
        """
        return 12, 12

    def init_beam_header(self):
        """Do nothing - beam header initialized in __init__ for this class."""
        pass


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
            # major axis North-South, along y
            assert U.isclose(
                b.kernel[mid + 5, mid],
                b.kernel[mid, mid + 10],
                rtol=1e-2,
            )
        elif bpa == 90 * U.deg:
            # major axis East-West, along x
            assert U.isclose(
                b.kernel[mid + 10, mid],
                b.kernel[mid, mid + 5],
                rtol=1e-2,
            )

    @pytest.mark.parametrize("bpa", (45 * U.deg, 135 * U.deg))
    def test_Gaussian_beam_position_angle(self, bpa):
        """
        Check that the beam position angle is measured East of North.

        Pixel x increases towards the West (decreasing RA), so a beam with position
        angle 45deg is elongated towards decreasing x and increasing y.
        """
        d = DataCube(
            n_px_x=1,
            n_px_y=1,
            n_channels=1,
            px_size=1 * U.arcsec,
            channel_width=1 * U.km / U.s,
        )
        b = GaussianBeam(bmaj=20 * U.arcsec, bmin=10 * U.arcsec, bpa=bpa, truncate=4)
        b.init_kernel(d)
        mid = b.kernel.shape[0] // 2
        north_east = b.kernel[mid - 5, mid + 5]
        north_west = b.kernel[mid + 5, mid + 5]
        if bpa == 45 * U.deg:
            assert north_east > 2 * north_west
        elif bpa == 135 * U.deg:
            assert north_west > 2 * north_east

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

    def test_kernel_orientation(self):
        """
        Check the orientation of the beam.

        Uses a kernel that is not symmetric to catch reflections and transpositions.
        The kernel is indexed (x, y) like the datacube. East is towards lower x (RA
        increases towards lower pixel index) and North is towards higher y.
        """
        d = DataCube(
            n_px_x=1,
            n_px_y=1,
            n_channels=1,
            px_size=1 * U.arcsec,
            channel_width=1 * U.km / U.s,
        )
        b = OffsetBeam(bmaj=3 * U.arcsec)
        b.init_kernel(d)
        mid_x, mid_y = b.kernel.shape[0] // 2, b.kernel.shape[1] // 2
        peak = np.unravel_index(np.argmax(b.kernel), b.kernel.shape)
        assert peak == (mid_x - 6, mid_y + 3)

    def test_non_square_kernel(self):
        """Check that x and y kernel sizes and padding are handled independently."""

        class NonSquareGaussianBeam(GaussianBeam):
            """A beam with differing kernel sizes along x and y."""

            def kernel_size_px(self) -> tuple[int, int]:
                return 3, 7

        d = DataCube(
            n_px_x=1,
            n_px_y=1,
            n_channels=1,
            px_size=1 * U.arcsec,
            channel_width=1 * U.km / U.s,
        )
        b = NonSquareGaussianBeam(bmaj=2 * U.arcsec)
        b.init_kernel(d)
        assert b.kernel.shape == (7, 15)
        assert b.needs_pad() == (3, 7)

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
