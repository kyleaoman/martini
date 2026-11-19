"""Tests of miscellaneous utilities."""

import pytest
import numpy as np
from astropy import units as U
from martini._util import _to_unit_in_place


class TestToUnitInPlace:
    """Test unit conversions that reuse memory."""

    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    def test_pixel_scaling(self, dtype):
        """Check an equivalency that is a pure scaling converts in place."""
        px_area = 4.0
        equivalencies = [
            (
                U.Jy * U.pix**-2,
                U.Jy * U.arcsec**-2,
                lambda x: x / px_area,
                lambda x: x * px_area,
            )
        ]
        q = U.Quantity(np.arange(10, dtype=dtype), U.Jy * U.pix**-2)
        expected = q.to(U.Jy * U.arcsec**-2, equivalencies=equivalencies)
        result = _to_unit_in_place(q, U.Jy * U.arcsec**-2, equivalencies)
        assert np.shares_memory(result, q)
        assert result.dtype == dtype
        assert U.allclose(result, expected)

    def test_beam_scaling(self):
        """Check a beam area equivalency converts in place."""
        equivalencies = U.beam_angular_area(50 * U.arcsec**2)
        q = U.Quantity(np.ones(5), U.Jy * U.arcsec**-2)
        expected = q.to(U.Jy * U.beam**-1, equivalencies=equivalencies)
        result = _to_unit_in_place(q, U.Jy * U.beam**-1, equivalencies)
        assert np.shares_memory(result, q)
        assert U.allclose(result, expected)

    def test_offset_falls_back(self):
        """Check a conversion with an offset falls back to a copy."""
        q = U.Quantity(np.array([0.0, 10.0]), U.deg_C)
        result = _to_unit_in_place(q, U.K, U.temperature())
        assert not np.shares_memory(result, q)
        assert U.allclose(result, [273.15, 283.15] * U.K)
