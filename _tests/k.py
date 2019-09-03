import numpy as np
import matplotlib.pyplot as pp
from martini.sph_kernels import WendlandC2Kernel, GaussianKernel, \
    CubicSplineKernel, WendlandC6Kernel
from astropy import units as U
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('k.pdf') as pdffile:
    for K in (GaussianKernel(truncate=2.), WendlandC2Kernel(),
              CubicSplineKernel(), WendlandC6Kernel()):
        pp.clf()
        r = np.linspace(0, 2, 50)
        pp.semilogy(r, K.eval_kernel(r, 1), '-k')
        pp.axhline(K.eval_kernel(0, 1) / 2)
        pp.axvline(K.fwhm / 2)
        pp.axvline(K.fwhm * K.size_in_fwhm())
        pp.savefig(pdffile, format='pdf')
        pp.clf()
        vmax = 100
        h = 50

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
            dij = np.vstack((xgrid[eval_grid], ygrid[eval_grid]))
            try:
                IKi = dr ** 2 * np.sum(K.kernel_integral(
                    dij * U.pix,
                    h * np.ones(rgrid.shape)[eval_grid].flatten() * U.pix
                ))
            except NotImplementedError:
                pass
            else:
                pp.plot(
                    ri / h,
                    IKi,
                    '.r',
                    zorder=0
                )

        r = np.linspace(0, 2, 50)
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
            pp.plot(
                ri,
                dr ** 3 * np.sum(K.eval_kernel(rgrid[eval_grid], 1)),
                'xb',
                zorder=-1
            )
        pp.savefig(pdffile, format='pdf')
