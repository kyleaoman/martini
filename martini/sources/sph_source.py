import numpy as np
from astropy.coordinates import (
    CartesianRepresentation,
    CartesianDifferential,
    SkyCoord,
    SpectralCoord,
    ICRS,
)
from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.units as U
from ._L_align import L_align
from ._cartesian_translation import translate, translate_d
from ..datacube import HIfreq

# Extend CartesianRepresentation to allow coordinate translation
setattr(CartesianRepresentation, "translate", translate)

# Extend CartesianDifferential to allow velocity (or other differential)
# translation
setattr(CartesianDifferential, "translate", translate_d)

_origin = CartesianRepresentation(
    np.zeros((3, 1)) * U.kpc,
    differentials={"s": CartesianDifferential(np.zeros((3, 1)) * U.km / U.s)},
)


class SPHSource(object):
    """
    Class abstracting HI emission sources consisting of SPH simulation
    particles.

    This class constructs an HI emission source from arrays of SPH particle
    properties: mass, smoothing length, temperature, position, and velocity.

    Parameters
    ----------
    distance : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Source distance, also used to set the velocity offset via Hubble's law.
        (Default: ``3 * U.Mpc``)

    vpeculiar : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of velocity.
        Source peculiar velocity, added to the velocity from Hubble's law.
        (Default: ``0 * U.km * U.s**-1``)

    rotation : dict, optional
        Must have a single key, which must be one of ``axis_angle``, ``rotmat`` or
        ``L_coords``. Note that the 'y-z' plane will be the one eventually placed in the
        plane of the "sky". The corresponding value must be:

        - ``axis_angle`` : 2-tuple, first element one of 'x', 'y', 'z' for the \
        axis to rotate about, second element a :class:`~astropy.units.Quantity` with \
        dimensions of angle, indicating the angle to rotate through.
        - ``rotmat`` : A (3, 3) :class:`~numpy.ndarray` specifying a rotation.
        - ``L_coords`` : A 2-tuple containing an inclination and an azimuthal \
        angle (both :class:`~astropy.units.Quantity` instances with dimensions of \
        angle). The routine will first attempt to identify a preferred plane \
        based on the angular momenta of the central 1/3 of particles in the \
        source. This plane will then be rotated to lie in the plane of the \
        "sky" ('y-z'), rotated by the azimuthal angle about its angular \
        momentum pole (rotation about 'x'), and inclined (rotation about \
        'y'). A 3-tuple may be provided instead, in which case the third \
        value specifies the position angle on the sky (second rotation about 'x'). \
        The default position angle is 270 degrees.

        (Default: ``np.eye(3)``)

    ra : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Right ascension for the source centroid. (Default: ``0 * U.deg``)

    dec : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Declination for the source centroid. (Default: ``0 * U.deg``)

    h : float, optional
        Dimensionless hubble constant,
        :math:`H_0 = h (100\\,\\mathrm{km}\\,\\mathrm{s}^{-1}\\,\\mathrm{Mpc}^{-1})`.
        (Default: ``0.7``)

    T_g : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of temperature.
        Particle temperature.

    mHI_g : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of mass.
        Particle HI mass.

    xyz_g : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity` , with dimensions of length.
        Particle position offset from source centroid. Note that the 'y-z'
        plane is that eventually placed in the plane of the "sky"; 'x' is
        the axis corresponding to the "line of sight".

    vxyz_g : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of velocity.
        Particle velocity offset from source centroid. Note that the 'y-z'
        plane is that eventually placed in the plane of the "sky"; 'x' is
        the axis corresponding to the "line of sight".

    hsm_g : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Particle SPH smoothing lengths, defined as the FWHM of the smoothing kernel.
        Smoothing lengths are variously defined in the literature as the radius where
        the kernel amplitude reaches 0, or some rational fraction of this radius (and
        other definitions may well exist). The FWHM requested here is not a standard
        choice (with the exception of SWIFT snapshots!), but has the advantage of avoiding
        ambiguity in the definition.

    coordinate_axis: int, optional
        Rank of axis corresponding to position or velocity of a single
        particle. I.e. ``coordinate_axis=0`` if shape is (3, N), or ``1`` if (N, 3).
        Usually prefer to omit this as it can be determined automatically, but is
        ambiguous for sources with exactly 3 particles. (Default: ``None``)

    coordinate_frame : ~astropy.coordinates.builtin_frames.baseradec.BaseRADecFrame, \
    optional
        The coordinate frame assumed in converting particle coordinates to RA and Dec, and
        for transforming coordinates and velocities to the data cube frame. The frame
        needs to have a well-defined velocity as well as spatial origin. Recommended
        frames are :class:`~astropy.coordinates.GCRS`, :class:`~astropy.coordinates.ICRS`,
        :class:`~astropy.coordinates.HCRS`, :class:`~astropy.coordinates.LSRK`,
        :class:`~astropy.coordinates.LSRD` or :class:`~astropy.coordinates.LSR`. The frame
        should be passed initialized, e.g. ``ICRS()`` (not just ``ICRS``).
        (Default: ``astropy.coordinates.ICRS()``)
    """

    def __init__(
        self,
        distance=3.0 * U.Mpc,
        vpeculiar=0.0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        h=0.7,
        T_g=None,
        mHI_g=None,
        xyz_g=None,
        vxyz_g=None,
        hsm_g=None,
        coordinate_axis=None,
        coordinate_frame=ICRS(),
    ):
        if coordinate_axis is None:
            if (xyz_g.shape[0] == 3) and (xyz_g.shape[1] != 3):
                coordinate_axis = 0
            elif (xyz_g.shape[0] != 3) and (xyz_g.shape[1] == 3):
                coordinate_axis = 1
            elif xyz_g.shape == (3, 3):
                raise RuntimeError(
                    "martini.sources.SPHSource: cannot guess "
                    "coordinate_axis with shape (3, 3), provide"
                    " explicitly."
                )
            else:
                raise RuntimeError(
                    "martini.sources.SPHSource: incorrect "
                    "coordinate shape (not (3, N) or (N, 3))."
                )
        else:
            coordinate_axis = coordinate_axis

        if xyz_g.shape != vxyz_g.shape:
            raise ValueError(
                "martini.sources.SPHSource: xyz_g and vxyz_g must"
                " have matching shapes."
            )

        if coordinate_axis == 0:
            xyz_g = xyz_g.T
            vxyz_g = vxyz_g.T
        self.coordinate_frame = coordinate_frame
        self.h = h
        self.T_g = T_g
        self.mHI_g = mHI_g
        self.input_mass = mHI_g.sum()
        self.coordinates_g = CartesianRepresentation(
            xyz_g,
            xyz_axis=1,
            differentials={"s": CartesianDifferential(vxyz_g, xyz_axis=1)},
        )
        self.hsm_g = hsm_g

        self.npart = self.coordinates_g.size

        self.ra = ra
        self.dec = dec
        self.distance = distance
        self.vpeculiar = vpeculiar
        self.vhubble = (self.h * 100.0 * U.km * U.s**-1 * U.Mpc**-1) * self.distance
        self.vsys = self.vhubble + self.vpeculiar
        self.current_rotation = np.eye(3)
        self.rotate(**rotation)
        self.skycoords = None
        self.spectralcoords = None
        self.pixcoords = None
        return

    def _init_skycoords(self, _reset=True):
        # _reset False only for unit testing
        direction_vector = np.array(
            [
                np.cos(self.ra) * np.cos(self.dec),
                np.sin(self.ra) * np.cos(self.dec),
                np.sin(self.dec),
            ]
        )
        distance_vector = direction_vector * self.distance
        vsys_vector = direction_vector * self.vsys
        self.rotate(axis_angle=("y", -self.dec))
        self.rotate(axis_angle=("z", self.ra))
        self.translate(distance_vector)
        self.boost(vsys_vector)
        self.skycoords = SkyCoord(
            self.coordinates_g, frame=self.coordinate_frame, copy=True
        )
        origin_skycoord = SkyCoord(
            x=0 * U.kpc,
            y=0 * U.kpc,
            z=0 * U.kpc,
            v_x=0 * U.km / U.s,
            v_y=0 * U.km / U.s,
            v_z=0 * U.km / U.s,
            representation_type="cartesian",
            differential_type="cartesian",
            frame=self.coordinate_frame,
        )
        self.spectralcoords = SpectralCoord(
            self.skycoords.radial_velocity,
            doppler_convention="radio",
            doppler_rest=HIfreq,
            target=self.skycoords,
            observer=origin_skycoord,
        )
        if _reset:
            self.boost(-vsys_vector)
            self.translate(-distance_vector)
            self.rotate(axis_angle=("z", -self.ra))
            self.rotate(axis_angle=("y", self.dec))
        return

    def _init_pixcoords(self, datacube, origin=0):
        skycoords_df_frame = self.skycoords.transform_to(datacube.coordinate_frame)
        spectralcoords_df_specsys = (
            self.spectralcoords.with_observer_stationary_relative_to(datacube.specsys)
        )
        # pixels indexed from 0 (not like in FITS!) for better use with numpy
        self.pixcoords = (
            np.vstack(
                datacube.wcs.sub(3).wcs_world2pix(
                    skycoords_df_frame.ra.to(datacube.units[0]),
                    skycoords_df_frame.dec.to(datacube.units[1]),
                    spectralcoords_df_specsys.to(datacube.units[2]),
                    origin,
                )
            )
            * U.pix
        )
        return

    def apply_mask(self, mask):
        """
        Remove particles from source arrays according to a mask.

        Parameters
        ----------
        mask : ~numpy.typing.ArrayLike
            Boolean mask. Remove particles with indices corresponding to
            ``False`` values from the source arrays.
        """

        if mask.size != self.npart:
            raise ValueError("Mask must have same length as particle arrays.")
        mask_sum = np.sum(mask)
        if mask_sum == 0:
            raise RuntimeError("No source particles in target region.")
        self.npart = mask_sum
        if not self.T_g.isscalar:
            self.T_g = self.T_g[mask]
        if not self.mHI_g.isscalar:
            self.mHI_g = self.mHI_g[mask]
        self.coordinates_g = self.coordinates_g[mask]
        if self.skycoords is not None:
            self.skycoords = self.skycoords[mask]
        if self.spectralcoords is not None:
            self.spectralcoords = self.spectralcoords[mask]
        if self.pixcoords is not None:
            self.pixcoords = self.pixcoords[:, mask]
        if not self.hsm_g.isscalar:
            self.hsm_g = self.hsm_g[mask]
        return

    def rotate(self, axis_angle=None, rotmat=None, L_coords=None):
        """
        Rotate the source.

        The arguments correspond to different rotation types. Multiple types cannot be
        given in a single function call.

        Parameters
        ----------
        axis_angle : tuple
            First element one of {``"x"``, ``"y"``, ``"z"``} for the axis to rotate about,
            second element a :class:`~astropy.units.Quantity` with dimensions of angle,
            indicating the angle to rotate through (right-handed rotation).
        rotmat : ~numpy.typing.ArrayLike
            Rotation matrix with shape (3, 3).
        L_coords : tuple
            First element containing an inclination, second element an
            azimuthal angle (both :class:`~astropy.units.Quantity` instances with
            dimensions of angle). The routine will first attempt to identify
            a preferred plane based on the angular momenta of the central 1/3
            of particles in the source. This plane will then be rotated to lie
            in the 'y-z' plane, followed by a rotation by the azimuthal angle
            about its angular momentum pole (rotation about 'x'), and then
            inclined (rotation about 'y'). By default the position angle on the
            sky is 270 degrees, but if a third element is provided it sets the
            position angle (second rotation about 'x').
        """

        args_given = (axis_angle is not None, rotmat is not None, L_coords is not None)
        if np.sum(args_given) == 0:
            # no-op
            return
        elif np.sum(args_given) > 1:
            raise ValueError("Multiple rotations in a single call not allowed.")

        do_rot = np.eye(3)

        if axis_angle is not None:
            # rotation_matrix gives left-handed rotation, so transpose for right-handed
            do_rot = rotation_matrix(axis_angle[1], axis=axis_angle[0]).T.dot(do_rot)

        if rotmat is not None:
            do_rot = rotmat.dot(do_rot)

        if L_coords is not None:
            if len(L_coords) == 2:
                incl, az_rot = L_coords
                pa = 270 * U.deg
            elif len(L_coords) == 3:
                incl, az_rot, pa = L_coords
            do_rot = L_align(
                self.coordinates_g.get_xyz(),
                self.coordinates_g.differentials["s"].get_d_xyz(),
                self.mHI_g,
                frac=0.3,
                Laxis="x",
            ).dot(do_rot)
            # rotation_matrix gives left-handed rotation, so transpose for right-handed
            do_rot = rotation_matrix(az_rot, axis="x").T.dot(do_rot)
            do_rot = rotation_matrix(incl, axis="y").T.dot(do_rot)
            if incl >= 0:
                do_rot = rotation_matrix(pa - 90 * U.deg, axis="x").T.dot(do_rot)
            else:
                do_rot = rotation_matrix(pa - 270 * U.deg, axis="x").T.dot(do_rot)

        self.current_rotation = do_rot.dot(self.current_rotation)
        self.coordinates_g = self.coordinates_g.transform(do_rot)
        return

    def translate(self, translation_vector):
        """
        Translate the source.

        Note that the "line of sight" is along the 'x' axis.

        Parameters
        ----------
        translation_vector : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with shape (3, ), with dimensions of length.
            Vector by which to offset the source particle coordinates.
        """

        self.coordinates_g = self.coordinates_g.translate(translation_vector)
        return

    def boost(self, boost_vector):
        """
        Apply an offset to the source velocity.

        Note that the "line of sight" is along the 'x' axis.

        Parameters
        ----------
        translation_vector : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with shape (3, ), with dimensions of
            velocity.
            Vector by which to offset the source particle velocities.
        """

        self.coordinates_g.differentials["s"] = self.coordinates_g.differentials[
            "s"
        ].translate(boost_vector)
        return

    def save_current_rotation(self, fname):
        """
        Output current rotation matrix to file.

        This includes the rotations applied for RA and Dec. The rotation matrix can be
        applied to astropy coordinates (e.g. a
        :class:`~astropy.coordinates.representation.cartesian.CartesianRepresentation`) as
        ``coordinates.transform(np.loadtxt(fname))``.

        Parameters
        ----------
        fname : str
            File in which to save rotation matrix. A file handle can also be passed.
        """

        np.savetxt(fname, self.current_rotation)
        return

    def preview(
        self,
        max_points=5000,
        fig=1,
        lim=None,
        vlim=None,
        point_scaling="auto",
        title="",
        save=None,
    ):
        """
        Produce a figure showing the source particle coordinates and velocities.

        Makes a 3-panel figure showing the projection of the source as it will appear in
        the mock observation. The first panel shows the particles in the y-z plane,
        coloured by the x-component of velocity (MARTINI projects the source along the
        x-axis). The second and third panels are position-velocity diagrams showing the
        x-component of velocity against the y and z coordinates, respectively.

        Parameters
        ----------
        max_points : int, optional
            Maximum number of points to draw per panel, the particles will be randomly
            subsampled if the source has more. (Default: ``1000``)

        fig : int, optional
            Number of the figure in matplotlib, it will be created as ``plt.figure(fig)``.
            (Default: ``1``)

        lim : ~astropy.units.Quantity, optional
            :class:`~astropy.units.Quantity`, with dimensions of length.
            The coordinate axes extend from -lim to lim. If unspecified, the maximum
            absolute coordinate of particles in the source is used. (Default: ``None``)

        vlim : ~astropy.units.Quantity, optional
            :class:`~astropy.units.Quantity`, with dimensions of speed.
            The velocity axes and colour bar extend from ``-vlim`` to ``vlim``. If
            unspecified, the maximum absolute velocity of particles in the source is used.
            (Default: ``None``)

        point_scaling : str, optional
            By default points are scaled in size and transparency according to their HI
            mass and the smoothing length (loosely proportional to their surface
            densities, but with different scaling to achieve a visually useful plot). For
            some sources the automatic scaling may not give useful results, using
            ``point_scaling="fixed"`` will plot points of constant size without opacity.
            (Default: ``"auto"``)

        title : str, optional
            A title for the figure can be provided. (Default: ``""``)

        save : str, optional
            If provided, the figure is saved using ``plt.savefig(save)``. A ``.png`` or
            ``.pdf`` suffix is recommended. (Default: ``None``)

        Returns
        -------
        out : ~matplotlib.figure.Figure
            The preview figure.
        """
        import matplotlib.pyplot as plt

        # every Nth particle to plot at most max_points, or all particles
        lim = (
            max(
                np.max(np.abs(self.coordinates_g.y.to_value(U.kpc))),
                np.max(np.abs(self.coordinates_g.z.to_value(U.kpc))),
            )
            if lim is None
            else lim.to_value(U.kpc)
        )
        vlim = (
            np.max(
                np.abs(self.coordinates_g.differentials["s"].d_x.to_value(U.km / U.s))
            )
            if vlim is None
            else vlim.to_value(U.km / U.s)
        )
        cmask = np.logical_and.reduce(
            (
                np.abs(self.coordinates_g.y.to_value(U.kpc)) < lim,
                np.abs(self.coordinates_g.z.to_value(U.kpc)) < lim,
                np.abs(self.coordinates_g.differentials["s"].d_x.to_value(U.km / U.s))
                < vlim,
            )
        )
        nparts = cmask.sum()
        mask = np.arange(self.mHI_g.size)[cmask][:: max(nparts // max_points, 1)]
        # mass_factor = (
        #     1
        #     if (self.mHI_g.isscalar or mask.size <= 1)
        #     else (self.mHI_g[mask] / self.mHI_g[mask].max()).to_value(
        #         U.dimensionless_unscaled
        #     )
        # )
        hsm_factor = (
            1
            if (self.hsm_g.isscalar or mask.size <= 1)
            else (1 - (self.hsm_g[mask] / self.hsm_g[mask].max()) ** 0.1).to_value(
                U.dimensionless_unscaled
            )
        )
        alpha = hsm_factor if point_scaling == "auto" else 1.0
        size_scale = (
            self.hsm_g.to_value(U.kpc) / lim
            if (self.hsm_g.isscalar or mask.size <= 1)
            else (self.hsm_g[mask].to_value(U.kpc) / lim)
        )
        size = 300 * size_scale if point_scaling == "auto" else 10
        fig = plt.figure(fig, figsize=(12, 4))
        fig.clf()
        fig.suptitle(title)

        # ----- MOMENT 1 -----
        sp1 = fig.add_subplot(1, 3, 1, aspect="equal")
        sp1.set_facecolor("#222222")
        scatter = sp1.scatter(
            self.coordinates_g.y[mask].to_value(U.kpc),
            self.coordinates_g.z[mask].to_value(U.kpc),
            c=self.coordinates_g.differentials["s"].d_x[mask].to_value(U.km / U.s),
            marker="o",
            cmap="coolwarm",
            edgecolor="None",
            s=size,
            alpha=alpha,
            vmin=-vlim,
            vmax=vlim,
            zorder=0,
        )
        sp1.plot([0], [0], marker="+", ls="None", mec="grey", ms=6, zorder=1)
        sp1.set_xlabel(r"$y\,[\mathrm{kpc}]$")
        sp1.set_ylabel(r"$z\,[\mathrm{kpc}]$")
        sp1.set_xlim((lim, -lim))
        sp1.set_ylim((-lim, lim))
        cb = fig.colorbar(mappable=scatter, ax=sp1, orientation="horizontal")
        cb.set_label(r"$v_x\,[\mathrm{km}\,\mathrm{s}^{-1}]$")

        # ----- PV Y -----
        sp2 = fig.add_subplot(1, 3, 2)
        sp2.set_facecolor("#222222")
        sp2.scatter(
            self.coordinates_g.y[mask].to_value(U.kpc),
            self.coordinates_g.differentials["s"].d_x[mask].to_value(U.km / U.s),
            c="white",
            edgecolors="None",
            marker="o",
            s=size,
            alpha=alpha,
            zorder=0,
        )
        sp2.plot([0], [0], marker="+", ls="None", mec="red", ms=6, zorder=1)
        sp2.set_xlim((lim, -lim))
        sp2.set_ylim((-vlim, vlim))
        sp2.set_xlabel(r"$y\,[\mathrm{kpc}]$")
        sp2.set_ylabel(r"$v_x\,[\mathrm{km}\,\mathrm{s}^{-1}]$")

        # ----- PV Z -----
        sp3 = fig.add_subplot(1, 3, 3)
        sp3.set_facecolor("#222222")
        sp3.scatter(
            self.coordinates_g.z[mask].to_value(U.kpc),
            self.coordinates_g.differentials["s"].d_x[mask].to_value(U.km / U.s),
            c="white",
            edgecolors="None",
            marker="o",
            s=size,
            alpha=alpha,
            zorder=0,
        )
        sp3.plot([0], [0], marker="+", ls="None", mec="red", ms=6, zorder=1)
        sp3.set_xlim((-lim, lim))
        sp3.set_ylim((-vlim, vlim))
        sp3.set_xlabel(r"$z\,[\mathrm{kpc}]$")
        sp3.set_ylabel(r"$v_x\,[\mathrm{km}\,\mathrm{s}^{-1}]$")

        fig.subplots_adjust(wspace=0.3)
        if save is not None:
            plt.savefig(save)
        return fig
