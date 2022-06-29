import pytest
import numpy as np
from math import isclose
from martini.sph_kernels import WendlandC2Kernel, GaussianKernel, \
    CubicSplineKernel, WendlandC6Kernel, DiracDeltaKernel, AdaptiveKernel
from astropy import units as U

# kernels that have a well-defined FWHM, i.e. not dirac-delta, adaptive
basic_kernels = WendlandC2Kernel, WendlandC6Kernel, CubicSplineKernel, GaussianKernel
all_kernels = basic_kernels + (DiracDeltaKernel, AdaptiveKernel)

for k in all_kernels:
    k.noFWHMwarn = True


class TestSPHKernels:

    @pytest.mark.parametrize("kernel", basic_kernels)
    def test_fwhm_is_one(self, kernel):
        """
        Check that value at FWHM is half of peak value.
        """
        k = kernel()
        fwhm = 1  # all kernels should be implemented s.t. this is true
        assert isclose(k.eval_kernel(fwhm / 2, 1),  k.eval_kernel(0, 1) / 2)

    @pytest.mark.parametrize("kernel", basic_kernels)
    def test_extent(self, kernel):
        """
        Check that kernel goes to zero at its stated size.
        """
        k = kernel()
        fwhm = 1  # all kernels should be implemented s.t. this is true
        assert k.eval_kernel(fwhm * k.size_in_fwhm + 1.e-5, 1) == 0
        assert k.eval_kernel(fwhm * k.size_in_fwhm - 1.e-5, 1) > 0

    @pytest.mark.parametrize("kernel", basic_kernels)
    def test_2D_integral(self, kernel):
        """
        Check numerically that integral of 3D kernel and 2D projection agree.
        """
        x_2d = list()
        y_2d = list()
        x_3d = list()
        y_3d = list()
        k = kernel()
        vmax = 50
        h = 25
        r = np.arange(0, vmax)
        rmid = .5 * (r[1:] + r[:-1])
        dr = r[1] - r[0]
        xgrid, ygrid = np.meshgrid(
            np.r_[r[1:][::-1], r],
            np.r_[r[1:][::-1], r]
        )
        rgrid = np.sqrt(
            np.power(xgrid, 2)
            + np.power(ygrid, 2)
        )
        for ri in rmid:
            eval_grid = rgrid <= ri
            k.sm_lengths = h * np.ones(rgrid.shape)[eval_grid].flatten() * U.pix
            dij = np.vstack((xgrid[eval_grid], ygrid[eval_grid]))
            IKi = dr ** 2 * np.sum(k.px_weight(
                dij * U.pix,
            ))
            x_2d.append(ri / h)
            y_2d.append(IKi.to_value(U.pix ** -2))

        r = r / h
        rmid = .5 * (r[1:] + r[:-1])
        dr = r[1] - r[0]
        xgrid, ygrid, zgrid = np.meshgrid(
            np.r_[r[1:][::-1], r],
            np.r_[r[1:][::-1], r],
            np.r_[r[1:][::-1], r]
        )
        rgrid = np.sqrt(
            np.power(xgrid, 2)
            + np.power(ygrid, 2)
            + np.power(zgrid, 2)
        )
        Rgrid = np.sqrt(
            np.power(xgrid, 2)
            + np.power(ygrid, 2)
        )
        for ri in rmid:
            eval_grid = Rgrid <= ri
            x_3d.append(ri)
            y_3d.append(dr ** 3 * np.sum(k.eval_kernel(rgrid[eval_grid], 1)))

        assert np.allclose(x_2d, x_3d)  # we evaluated integrals at same radii
        assert np.allclose(y_2d, y_3d, rtol=2.e-2)  # integrals within 2%
