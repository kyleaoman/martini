import pytest
import numpy as np
from martini import Martini, DataCube
from martini.spectral_models import GaussianSpectrum
from martini.sources import SPHSource
from martini.sph_kernels import (
    _WendlandC2Kernel,
    _GaussianKernel,
    _CubicSplineKernel,
    _QuarticSplineKernel,
    _WendlandC6Kernel,
    DiracDeltaKernel,
    _AdaptiveKernel,
    WendlandC2Kernel,
    GaussianKernel,
    CubicSplineKernel,
    QuarticSplineKernel,
    WendlandC6Kernel,
)
from astropy import units as U

# kernels that have a well-defined FWHM, i.e. not dirac-delta, adaptive
fwhm_kernels = (
    _WendlandC2Kernel,
    _WendlandC6Kernel,
    _CubicSplineKernel,
    _GaussianKernel,
    _QuarticSplineKernel,
)
simple_kernels = fwhm_kernels + (DiracDeltaKernel,)
recommended_kernels = (
    WendlandC2Kernel,
    WendlandC6Kernel,
    CubicSplineKernel,
    GaussianKernel,
    QuarticSplineKernel,
)
adaptive_kernels = recommended_kernels + (_AdaptiveKernel,)
all_kernels = simple_kernels + adaptive_kernels


def total_kernel_weight(k, h, ngrid=50):
    r = np.arange(0, ngrid)
    dr = (r[1] - r[0]) * U.pix
    xgrid, ygrid = np.meshgrid(np.r_[r[1:][::-1], r], np.r_[r[1:][::-1], r])
    dij = np.vstack((xgrid.flatten(), ygrid.flatten()))
    k.sm_lengths = np.ones(dij.shape[1]) * h * U.pix
    return (dr**2 * np.sum(k._px_weight(dij * U.pix))).to_value(
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
                k._px_weight(
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
        (
            _WendlandC2Kernel,
            _WendlandC6Kernel,
            _CubicSplineKernel,
            _QuarticSplineKernel,
        ),
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
            total_kernel_weight(k, kernel.min_valid_size * 1.001), 1.0, rtol=1.0e-2
        )
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
            total_kernel_weight(k, kernel.max_valid_size * 0.999), 1.0, rtol=1.0e-2
        )
        # check that with a larger size is more than 1% from 1.0
        if kernel not in (DiracDeltaKernel,):
            # DiracDeltaKernel will still converge to right answer, but is
            # a poor approximation on geometric grounds
            assert not np.isclose(
                total_kernel_weight(k, kernel.max_valid_size * 1.1), 1.0, rtol=1.0e-2
            )

    @pytest.mark.parametrize("truncate", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    def test_kernel_validation_Gaussian(self, truncate):
        """
        Check that our kernel integral approximations hold within the stated
        tolerance, in other words that we conserve mass within the same
        tolerance.
        """
        if truncate < 2.0:
            with pytest.raises(RuntimeError, match="with truncation <2sigma"):
                k = _GaussianKernel(truncate=truncate)
            return
        else:
            k = _GaussianKernel(truncate=truncate)
        # check that a very well-sampled case gives 1.0
        assert np.isclose(total_kernel_weight(k, 20), 1.0, rtol=3.0e-3)
        # check that the minimum size gives 1.0 within 1%
        assert np.isclose(
            total_kernel_weight(k, k.min_valid_size * 1.1), 1.0, rtol=1.0e-2
        )
        # don't check that with a smaller size is more than 1% from 1.0, off-centre
        # particle in pixel can fail even when this passes
        # assert not np.isclose(
        #     total_kernel_weight(k, k.min_valid_size * 0.9), 1.0, rtol=1.0e-2
        # )

    @pytest.mark.parametrize(
        ("kernel", "kernel_args"),
        (
            (_GaussianKernel, (2,)),
            (_GaussianKernel, (3,)),
            (_GaussianKernel, (4,)),
            (_GaussianKernel, (5,)),
            (_GaussianKernel, (6,)),
            (_WendlandC2Kernel, None),
            (_WendlandC6Kernel, None),
            (_CubicSplineKernel, None),
            (_QuarticSplineKernel, None),
            (DiracDeltaKernel, None),
        ),
    )
    def test_kernel_confirm_validation(self, kernel, kernel_args):
        """
        Setup a source at 3 Mpc and a datacube such that pixel scale is 1kpc. Provide
        smoothing lengths just above and below the minimum or maximum size and check that
        validation passes/fails accordingly.
        """
        if kernel_args is not None:

            def k():
                return kernel(*kernel_args)

        else:

            def k():
                return kernel()

        if hasattr(k(), "min_valid_size"):
            testcases = ((True, 0.9), (False, 1.1))
            threshold = k().min_valid_size
        elif hasattr(k(), "max_valid_size"):
            testcases = ((True, 1.1), (False, 0.9))
            threshold = k().max_valid_size
        for raises, factor in testcases:
            source = SPHSource(
                T_g=np.ones(1) * 1.0e4 * U.K,
                mHI_g=np.ones(1) * 1.0e4 * U.Msun,
                xyz_g=np.ones((1, 3)) * 1.0e-3 * U.kpc,
                vxyz_g=np.zeros((1, 3)) * U.km * U.s**-1,
                hsm_g=np.ones(1) * threshold * factor * U.kpc,
                distance=3 * U.Mpc,
            )
            datacube = DataCube(
                n_px_x=100,
                n_px_y=100,
                n_channels=32,
                px_size=68.75493542 * U.arcsec,
                channel_width=10 * U.km / U.s,
                spectral_centre=source.vsys,
            )
            m = Martini(
                source=source,
                datacube=datacube,
                beam=None,
                noise=None,
                spectral_model=GaussianSpectrum(),
                sph_kernel=k(),
                quiet=True,
            )
            if raises:
                with pytest.raises(RuntimeError, match="use this with care"):
                    m.sph_kernel._confirm_validation(quiet=True)
            else:
                m.sph_kernel._confirm_validation(quiet=True)  # should not raise


class TestAdaptiveKernels:
    @pytest.mark.parametrize(
        "kernel",
        (
            WendlandC2Kernel,
            WendlandC6Kernel,
            CubicSplineKernel,
            QuarticSplineKernel,
        ),
    )
    def test_kernel_selection(
        self, kernel, adaptive_kernel_test_source, adaptive_kernel_test_datacube
    ):
        """
        Test that the kernel is chosen correctly from those available.

        The smoothing lengths are respectively: (3.0, 1.0, 0.55, 0.1) kpc. Normally
        the first two should use the preferred kernel, the third should fall back to
        a `_GaussianKernel` with a large truncation radius, and the last should fall back
        to a `DiracDeltaKernel`. Assumes 1kpc pixels, which is what we'll use for testing.
        Note that the minimum valid size is compared to the smoothing length times the
        rescaling parameter, so for example
        `_WendlandC2Kernel.min_valid_size / _WendlandC2Kernel()._rescale` has a value of
        0.9477, so with our 1kpc pixels, a value in the smoothing length array of
        at least 0.9477kpc will use the preferred kernel, while less than 0.5kpc will
        use the `DiracDeltaKernel` and between these two will use the `_GaussianKernel`
        with truncation parameter of 6.
        """
        source = adaptive_kernel_test_source()
        source._init_skycoords()
        k = kernel()
        k._init_sm_lengths(source=source, datacube=adaptive_kernel_test_datacube)
        assert all(k.kernel_indices == np.array([0, 0, 2, 1]))

    def test_kernel_selection_Gaussian(
        self, adaptive_kernel_test_source, adaptive_kernel_test_datacube
    ):
        """
        Test that the kernel is chosen correctly from those available.

        The smoothing lengths are respectively (3.0, 1.0, 0.55, 0.1) kpc. For a
        `GaussianKernel` with truncation parameter of 3, the first should use the
        preferred kernel, the second and third should fall back to a `_GaussianKernel`
        with a truncation parameter of 6, and the last should fall back to a
        `DiracDeltaKernel`. The `_GaussianKernel` has a rescaling parameter of 1.
        """
        source = adaptive_kernel_test_source()
        source._init_skycoords()
        k = GaussianKernel(truncate=3.0)
        k._init_sm_lengths(source=source, datacube=adaptive_kernel_test_datacube)
        assert all(k.kernel_indices == np.array([0, 2, 2, 1]))

    @pytest.mark.parametrize("kernel", recommended_kernels)
    def test_raw_kernel_function(self, kernel):
        """
        Test that evaluating the `kernel` function gives the evaluation of the adaptive
        kernel's preferred kernel's `kernel` function.
        """
        eval_at = np.linspace(0, 3, 51)
        assert np.allclose(
            kernel().kernel(eval_at), kernel().kernels[0].kernel(eval_at)
        )

    @pytest.mark.parametrize("kernel", recommended_kernels)
    def test_eval_kernel_function(self, kernel):
        """
        Test that evaluating the `eval_kernel` function gives the evaluation of the
        adaptive kernel's preferred kernel's `eval_kernel` function.
        """
        eval_at = np.linspace(0, 3, 51)
        assert np.allclose(
            kernel().eval_kernel(eval_at, np.ones(eval_at.shape)),
            kernel().kernels[0].eval_kernel(eval_at, np.ones(eval_at.shape)),
        )
