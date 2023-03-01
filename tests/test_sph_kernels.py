import pytest
import numpy as np
from martini.sph_kernels import (
    WendlandC2Kernel,
    GaussianKernel,
    CubicSplineKernel,
    QuarticSplineKernel,
    WendlandC6Kernel,
    DiracDeltaKernel,
    AdaptiveKernel,
)
from astropy import units as U

# kernels that have a well-defined FWHM, i.e. not dirac-delta, adaptive
fwhm_kernels = (
    WendlandC2Kernel,
    WendlandC6Kernel,
    CubicSplineKernel,
    GaussianKernel,
    QuarticSplineKernel,
)
simple_kernels = fwhm_kernels + (DiracDeltaKernel,)
all_kernels = simple_kernels + (AdaptiveKernel,)

for k in all_kernels:
    k.noFWHMwarn = True


def total_kernel_weight(k, h, ngrid=50):
    r = np.arange(0, ngrid)
    dr = (r[1] - r[0]) * U.pix
    xgrid, ygrid = np.meshgrid(np.r_[r[1:][::-1], r], np.r_[r[1:][::-1], r])
    dij = np.vstack((xgrid.flatten(), ygrid.flatten()))
    k.sm_lengths = np.ones(dij.shape[1]) * h * U.pix
    return (dr**2 * np.sum(k.px_weight(dij * U.pix))).to_value(
        U.dimensionless_unscaled
    )


class TestSPHKernels:
    @pytest.mark.parametrize("kernel", fwhm_kernels)
    def test_fwhm_is_one(self, kernel):
        """
        Check that value at FWHM is half of peak value.
        """
        k = kernel()
        fwhm = 1  # all kernels should be implemented s.t. this is true
        assert np.isclose(k.eval_kernel(fwhm / 2, 1), k.eval_kernel(0, 1) / 2)

    @pytest.mark.parametrize("kernel", fwhm_kernels)
    def test_extent(self, kernel):
        """
        Check that kernel goes to zero at its stated size.
        """
        k = kernel()
        fwhm = 1  # all kernels should be implemented s.t. this is true
        assert k.eval_kernel(fwhm * k.size_in_fwhm + 1.0e-5, 1) == 0
        assert k.eval_kernel(fwhm * k.size_in_fwhm - 1.0e-5, 1) > 0

    @pytest.mark.parametrize("kernel", fwhm_kernels)
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
        rmid = 0.5 * (r[1:] + r[:-1])
        dr = r[1] - r[0]
        xgrid, ygrid = np.meshgrid(np.r_[r[1:][::-1], r], np.r_[r[1:][::-1], r])
        rgrid = np.sqrt(np.power(xgrid, 2) + np.power(ygrid, 2))
        for ri in rmid:
            eval_grid = rgrid <= ri
            k.sm_lengths = h * np.ones(rgrid.shape)[eval_grid].flatten() * U.pix
            dij = np.vstack((xgrid[eval_grid], ygrid[eval_grid]))
            IKi = dr**2 * np.sum(
                k.px_weight(
                    dij * U.pix,
                )
            )
            x_2d.append(ri / h)
            y_2d.append(IKi.to_value(U.pix**-2))

        r = r / h
        rmid = 0.5 * (r[1:] + r[:-1])
        dr = r[1] - r[0]
        xgrid, ygrid, zgrid = np.meshgrid(
            np.r_[r[1:][::-1], r], np.r_[r[1:][::-1], r], np.r_[r[1:][::-1], r]
        )
        rgrid = np.sqrt(np.power(xgrid, 2) + np.power(ygrid, 2) + np.power(zgrid, 2))
        Rgrid = np.sqrt(np.power(xgrid, 2) + np.power(ygrid, 2))
        for ri in rmid:
            eval_grid = Rgrid <= ri
            x_3d.append(ri)
            y_3d.append(dr**3 * np.sum(k.eval_kernel(rgrid[eval_grid], 1)))

        assert np.allclose(x_2d, x_3d)  # we evaluated integrals at same radii
        assert np.allclose(y_2d, y_3d, rtol=2.0e-2)  # integrals within 2%

    @pytest.mark.parametrize(
        "kernel",
        (WendlandC2Kernel, WendlandC6Kernel, CubicSplineKernel, QuarticSplineKernel),
    )
    def test_kernel_validation_minsize(self, kernel):
        """
        Check that our kernel integral approximations hold within the stated
        tolerance, in other words that we conserve mass within the same
        tolerance.
        """
        k = kernel()
        # check that a very well-sampled case gives 1.0
        assert np.isclose(total_kernel_weight(k, 20), 1.0, rtol=1.0e-3)
        # check that the minimum size gives 1.0 within 1%
        assert np.isclose(
            total_kernel_weight(k, kernel.min_valid_size), 1.0, rtol=1.0e-2
        )
        # check that with a smaller size is more than 1% from 1.0
        assert not np.isclose(
            total_kernel_weight(k, kernel.min_valid_size * 0.9), 1.0, rtol=1.0e-2
        )

    @pytest.mark.parametrize("kernel", (DiracDeltaKernel,))
    def test_kernel_validation_maxsize(self, kernel):
        """
        Check that our kernel integral approximations hold within the stated
        tolerance, in other words that we conserve mass within the same
        tolerance.
        """
        k = kernel()
        # check that a very well-sampled case gives 1.0
        assert np.isclose(total_kernel_weight(k, 20), 1.0, rtol=1.0e-3)
        # check that the maximum size gives 1.0 within 1%
        assert np.isclose(
            total_kernel_weight(k, kernel.max_valid_size), 1.0, rtol=1.0e-2
        )
        # check that with a larger size is more than 1% from 1.0
        if kernel not in (DiracDeltaKernel,):
            # DiracDeltaKernel will still converge to right answer, but is
            # a poor approximation on geometric grounds
            assert not np.isclose(
                total_kernel_weight(k, kernel.max_valid_size * 1.1), 1.0, rtol=1.0e-2
            )

    @pytest.mark.parametrize("truncate", (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    def test_kernel_validation_Gaussian(self, truncate):
        """
        Check that our kernel integral approximations hold within the stated
        tolerance, in other words that we conserve mass within the same
        tolerance.
        """
        if truncate < 1.0:
            with pytest.raises(RuntimeError, match="with truncation <1sigma"):
                k = GaussianKernel(truncate=truncate)
            return
        elif truncate >= 6.0:
            with pytest.warns(UserWarning, match="with truncation >=6sigma"):
                k = GaussianKernel(truncate=truncate)
        else:
            k = GaussianKernel(truncate=truncate)
        # check that a very well-sampled case gives 1.0 within 1%
        assert np.isclose(total_kernel_weight(k, 20), 1.0, rtol=1.0e-2)
        if k.lims is not None:
            # check that the lower limit size gives 1.0 within 1%
            if k.lims[0] > 0:
                assert np.isclose(total_kernel_weight(k, k.lims[0]), 1.0, rtol=1.0e-2)
            # check that the upper limit size gives 1.0 within 1%
            assert np.isclose(total_kernel_weight(k, k.lims[1]), 1.0, rtol=1.0e-2)
            # check that 1.1 * lower limit and 0.9 * upper limit are forbidden
            assert 1.1 * k.lims[0] < k.lims[1]
            assert 0.9 * k.lims[1] > k.lims[0]
            # check that with these values we are more than 1% from 1.0
            if k.lims[0] > 0:
                assert not np.isclose(
                    total_kernel_weight(k, 1.1 * k.lims[0]), 1.0, rtol=1.0e-2
                )
            assert not np.isclose(
                total_kernel_weight(k, 0.9 * k.lims[1]), 1.0, rtol=1.0e-2
            )
        else:
            # check that quite a poorly-sampled case gives 1.0 within 1%
            assert np.isclose(total_kernel_weight(k, 0.1), 1.0, rtol=1.0e-2)
