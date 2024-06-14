import os
import pytest
import numpy as np
from astropy import units as U
from astropy.coordinates.matrix_utilities import rotation_matrix
from martini.datacube import DataCube
from martini.sources import SPHSource
from martini.sources._cartesian_translation import translate, translate_d
from martini.sources._L_align import L_align
from astropy.coordinates import CartesianRepresentation, CartesianDifferential

try:
    import matplotlib
except ImportError:
    have_matplotlib = False
else:
    have_matplotlib = True


class TestSourceUtilities:
    def test_L_align(self):
        """
        Test that L_align produces expected rotation matrices, including saving to file.
        """
        # set up 4 particles in x-y plane rotating right-handed about zhat
        _xyz = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        _vxyz = np.array([[0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]])
        # tile outwards to make a 4-armed "windmill"
        xyz = np.vstack((_xyz * np.arange(1, 201).reshape(200, 1, 1))) * U.kpc
        vxyz = np.vstack((_vxyz * np.arange(1, 201).reshape(200, 1, 1))) * U.km / U.s
        m = np.ones(xyz.shape[0]) * U.Msun
        # rotating zhat to align with zhat should stay aligned with zhat
        # (but might arbitrarily rotate x-y plane)
        assert np.allclose(
            np.array([[0, 0, 1]]).dot(L_align(xyz, vxyz, m, Laxis="z")),
            np.array([0, 0, 1]),
        )
        # rotating zhat to align with xhat should align with xhat
        assert np.allclose(
            np.array([[0, 0, 1]]).dot(L_align(xyz, vxyz, m, Laxis="x")),
            np.array([1, 0, 0]),
        )
        # rotating zhat to align with yhat should align with yhat
        assert np.allclose(
            np.array([[0, 0, 1]]).dot(L_align(xyz, vxyz, m, Laxis="y")),
            np.array([0, 1, 0]),
        )
        # and also check transposed cases
        assert np.allclose(
            L_align(xyz.T, vxyz.T, m, Laxis="z").dot(np.array([0, 0, 1])),
            np.array([0, 0, 1]),
        )
        assert np.allclose(
            L_align(xyz.T, vxyz.T, m, Laxis="x").dot(np.array([0, 0, 1])),
            np.array([1, 0, 0]),
        )
        assert np.allclose(
            L_align(xyz.T, vxyz.T, m, Laxis="y").dot(np.array([0, 0, 1])),
            np.array([0, 1, 0]),
        )
        # finally check that saving to file works
        testfile = "testsaverot.npy"
        try:
            rotmat = L_align(xyz, vxyz, m, saverot=testfile)
            saved_rotmat = np.load(testfile)
        finally:
            if os.path.exists(testfile):
                os.remove(testfile)
        assert np.allclose(rotmat, saved_rotmat)

    def test_translate(self):
        """
        Check that cartesian representation transforms correctly.
        """
        setattr(CartesianRepresentation, "translate", translate)
        cr = CartesianRepresentation(np.zeros(3) * U.kpc)
        translation = np.ones(3) * U.kpc
        cr_translated = cr.translate(translation)
        assert U.allclose(cr_translated.get_xyz(), translation)

    def test_translate_d(self):
        """
        Check that cartesian differential transforms correctly.
        """
        setattr(CartesianDifferential, "translate", translate_d)
        cd = CartesianDifferential(np.zeros(3) * U.km / U.s)
        translation = np.ones(3) * U.km / U.s
        cd_translated = cd.translate(translation)
        assert U.allclose(cd_translated.get_d_xyz(), translation)


class TestSPHSource:
    def test_coordinate_input(self):
        """
        Check that different input shapes for coordinates give expected behaviour.
        """
        mHI_g = np.zeros(4) * U.Msun
        row_coords = np.zeros((4, 3)) * U.kpc
        row_vels = np.zeros((4, 3)) * U.km / U.s
        s = SPHSource(xyz_g=row_coords, vxyz_g=row_vels, mHI_g=mHI_g)
        assert s.coordinates_g.shape == (4,)
        assert s.coordinates_g.differentials["s"].shape == (4,)
        assert s.npart == 4
        col_coords = np.zeros((3, 4)) * U.kpc
        col_vels = np.zeros((3, 4)) * U.km / U.s
        s = SPHSource(xyz_g=col_coords, vxyz_g=col_vels, mHI_g=mHI_g)
        assert s.coordinates_g.shape == (4,)
        assert s.coordinates_g.differentials["s"].shape == (4,)
        assert s.npart == 4
        mHI_g = np.zeros(3) * U.Msun
        symm_coords = np.zeros((3, 3)) * U.kpc
        symm_coords[0, 1] = 1 * U.kpc
        symm_vels = np.zeros((3, 3)) * U.km / U.s
        symm_vels[0, 1] = 1 * U.km / U.s
        s = SPHSource(
            xyz_g=symm_coords, vxyz_g=symm_vels, mHI_g=mHI_g, coordinate_axis=0
        )
        assert s.coordinates_g.shape == (3,)
        assert s.coordinates_g.differentials["s"].shape == (3,)
        assert s.coordinates_g.x[1] == 1 * U.kpc
        assert s.coordinates_g.differentials["s"].d_x[1] == 1 * U.km / U.s
        assert s.npart == 3
        s = SPHSource(
            xyz_g=symm_coords, vxyz_g=symm_vels, mHI_g=mHI_g, coordinate_axis=1
        )
        assert s.coordinates_g.shape == (3,)
        assert s.coordinates_g.shape == (3,)
        assert s.coordinates_g.y[0] == 1 * U.kpc
        assert s.coordinates_g.differentials["s"].d_y[0] == 1 * U.km / U.s
        assert s.npart == 3
        with pytest.raises(RuntimeError, match="cannot guess coordinate_axis"):
            SPHSource(xyz_g=symm_coords, vxyz_g=symm_vels, mHI_g=mHI_g)
        with pytest.raises(ValueError, match="must have matching shapes"):
            SPHSource(xyz_g=row_coords, vxyz_g=col_vels, mHI_g=mHI_g)

    @pytest.mark.parametrize("ra", (0 * U.deg, 30 * U.deg, -30 * U.deg))
    @pytest.mark.parametrize("dec", (0 * U.deg, 30 * U.deg, -30 * U.deg))
    def test_ra_dec_rotation(self, ra, dec):
        """
        Check that coordinates rotate as needed before translation to observed position.
        """
        xyz_g = np.arange(3).reshape((1, 3)) * U.kpc
        vxyz_g = np.arange(3).reshape((1, 3)) * U.km / U.s
        mHI_g = np.zeros(3) * U.Msun
        s = SPHSource(
            xyz_g=xyz_g, vxyz_g=vxyz_g, mHI_g=mHI_g, ra=ra, dec=dec, distance=0 * U.Mpc
        )
        s._init_skycoords(_reset=False)
        R_y = np.array(
            [
                [np.cos(dec), 0, np.sin(dec)],
                [0, 1, 0],
                [-np.sin(dec), 0, np.cos(dec)],
            ]
        )
        R_z = np.array(
            [
                [np.cos(-ra), -np.sin(-ra), 0],
                [np.sin(-ra), np.cos(-ra), 0],
                [0, 0, 1],
            ]
        )
        rotmat = R_y.dot(R_z)
        assert U.allclose(s.coordinates_g.xyz.T, xyz_g.dot(rotmat))
        assert U.allclose(
            s.coordinates_g.differentials["s"].d_xyz.T, vxyz_g.dot(rotmat)
        )

    @pytest.mark.parametrize("ra", (0 * U.deg, 30 * U.deg, -30 * U.deg))
    @pytest.mark.parametrize("dec", (0 * U.deg, 30 * U.deg, -30 * U.deg))
    @pytest.mark.parametrize("distance", (0 * U.Mpc, 3 * U.Mpc))
    @pytest.mark.parametrize("vpeculiar", (0 * U.km / U.s, 100 * U.km / U.s))
    def test_dist_vpec_translation(self, ra, dec, distance, vpeculiar):
        """
        Check that coordinates translate correctly to observed position.
        """
        xyz_g = np.arange(3).reshape((1, 3)) * U.kpc
        vxyz_g = np.arange(3).reshape((1, 3)) * U.km / U.s
        mHI_g = np.zeros(3) * U.Msun
        s = SPHSource(
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            mHI_g=mHI_g,
            ra=ra,
            dec=dec,
            distance=distance,
            vpeculiar=vpeculiar,
        )
        s._init_skycoords(_reset=False)
        vsys = s.h * 100 * U.km / U.s / U.Mpc * distance + vpeculiar
        R_y = np.array(
            [
                [np.cos(dec), 0, np.sin(dec)],
                [0, 1, 0],
                [-np.sin(dec), 0, np.cos(dec)],
            ]
        )
        R_z = np.array(
            [
                [np.cos(-ra), -np.sin(-ra), 0],
                [np.sin(-ra), np.cos(-ra), 0],
                [0, 0, 1],
            ]
        )
        rotmat = R_y.dot(R_z)
        direction_vector = rotmat.T.dot(np.array([[1], [0], [0]]))
        assert U.allclose(
            s.coordinates_g.xyz.T, xyz_g.dot(rotmat) + direction_vector.T * distance
        )
        assert U.allclose(
            s.coordinates_g.differentials["s"].d_xyz.T,
            vxyz_g.dot(rotmat) + direction_vector.T * vsys,
        )

    def test_init_skycoords_resets(self, s):
        """
        Check that particle coordinate arrays are reset after initialising skycoords.
        """
        initial_coords = s.coordinates_g
        s._init_skycoords()
        ax_equal = [
            U.allclose(getattr(s.coordinates_g, ax), getattr(initial_coords, ax))
            for ax in "xyz"
        ]
        assert all(ax_equal)
        # make sure that it wasn't a no-op:
        s._init_skycoords(_reset=False)
        ax_equal = [
            U.allclose(getattr(s.coordinates_g, ax), getattr(initial_coords, ax))
            for ax in "xyz"
        ]
        assert not all(ax_equal)

    def test_init_pixcoords(self):
        """
        Check that pixel coordinates are accurately calculated from angular positions and
        velocity offsets.
        """
        # set distance so that 1kpc = 1arcsec
        distance = (1 * U.kpc / 1 / U.arcsec).to(U.Mpc, U.dimensionless_angles())
        # line up particles 1 per 1kpc = 1arcsec interval in RA and Dec
        # and 1 per 1 km / s interval in vlos
        # set h=0 so that velocity stays centred at 0
        source = SPHSource(
            distance=distance,
            h=0.0,
            T_g=np.ones(5) * 1e4 * U.K,
            mHI_g=np.ones(5) * 1e4 * U.Msun,
            xyz_g=U.Quantity(
                np.vstack(
                    (
                        np.zeros(6),
                        np.linspace(-2.5, 2.5, 6),
                        (np.linspace(-2.5, 2.5, 6)),
                    )
                ).T,
                U.kpc,
            ),
            vxyz_g=U.Quantity(
                np.vstack((np.linspace(-2.5, 2.5, 6), np.zeros(6), np.zeros(6))).T,
                U.km / U.s,
            ),
            hsm_g=np.ones(6) * U.kpc,
        )
        datacube = DataCube(
            n_px_x=6,
            n_px_y=6,
            n_channels=6,
            px_size=1 * U.arcsec,
            channel_width=1 * U.km / U.s,
        )
        expected_coords = (
            np.vstack((np.arange(6)[::-1], np.arange(6), np.arange(6)[::-1])) * U.pix
        )
        source._init_skycoords()
        source._init_pixcoords(datacube)
        assert U.allclose(
            source.pixcoords,
            expected_coords,
            atol=1e-4 * U.pix,
        )

    @pytest.mark.parametrize("ra", (0 * U.deg, 30 * U.deg, -30 * U.deg))
    @pytest.mark.parametrize("dec", (0 * U.deg, 30 * U.deg, -30 * U.deg))
    def test_sky_location(self, ra, dec):
        """
        Check that particle ends up where expected on the sky.
        """
        from astropy.coordinates import Angle

        xyz_g = np.zeros((1, 3)) * U.kpc
        vxyz_g = np.zeros((1, 3)) * U.km / U.s
        mHI_g = np.zeros(1) * U.Msun
        s = SPHSource(xyz_g=xyz_g, vxyz_g=vxyz_g, ra=ra, dec=dec, mHI_g=mHI_g)
        s._init_skycoords()
        assert U.isclose(s.skycoords.ra[0], Angle(ra).wrap_at(360 * U.deg))
        assert U.isclose(s.skycoords.dec[0], Angle(dec).wrap_at(180 * U.deg))

    def test_apply_mask(self, s):
        """
        Check that particle arrays can be masked.
        """
        particle_fields = ("T_g", "mHI_g", "hsm_g")
        npart_before_mask = s.T_g.size
        particles_before = {k: getattr(s, k) for k in particle_fields}
        particles_before["coordinates_g"] = s.coordinates_g
        # make sure we have a scalar value for one field to check it's handled
        assert particles_before["hsm_g"].isscalar
        mask = np.r_[
            np.ones(npart_before_mask - npart_before_mask // 3, dtype=int),
            np.zeros(npart_before_mask // 3, dtype=int),
        ]
        s.apply_mask(mask)
        for k in particle_fields:
            if not particles_before[k].isscalar:
                assert U.allclose(particles_before[k][mask], getattr(s, k))
            else:
                assert U.isclose(particles_before[k], getattr(s, k))
        assert U.allclose(
            particles_before["coordinates_g"].get_xyz()[:, mask],
            s.coordinates_g.get_xyz(),
        )
        assert U.allclose(
            particles_before["coordinates_g"].differentials["s"].get_d_xyz()[:, mask],
            s.coordinates_g.differentials["s"].get_d_xyz(),
        )

    def test_apply_badmask(self, s):
        """
        Check that bad masks are rejected.
        """
        with pytest.raises(
            ValueError, match="Mask must have same length as particle arrays."
        ):
            s.apply_mask(np.array([1, 2, 3]))
        with pytest.raises(RuntimeError, match="No source particles in target region."):
            s.apply_mask(np.zeros(s.npart, dtype=int))

    def test_rotate_axis_angle(self, s):
        """
        Test that we can rotate by an axis-angle transformation.
        """
        assert np.allclose(s.current_rotation, np.eye(3))
        axis = "z"
        angle = 30 * U.deg
        # expect a right-handed rotation:
        rotmat = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        s.rotate(axis_angle=(axis, angle))
        assert np.allclose(s.current_rotation, rotmat)

    def test_rotate_rotmat(self, s):
        """
        Check that we can rotate by a rotmat transformation.
        """
        assert np.allclose(s.current_rotation, np.eye(3))
        angle = 30 * U.deg
        rotmat = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        vector_before = s.coordinates_g.get_xyz()[:, 0]
        assert any(np.abs(vector_before) > 0)
        s.rotate(rotmat=rotmat)
        assert U.allclose(
            s.coordinates_g.get_xyz()[:, 0], np.dot(rotmat, vector_before)
        )

    @pytest.mark.parametrize("incl", (0 * U.deg, 30 * U.deg, -30 * U.deg))
    @pytest.mark.parametrize("az_rot", (0 * U.deg, 30 * U.deg, -30 * U.deg))
    @pytest.mark.parametrize("pa", (270 * U.deg, 300 * U.deg, 240 * U.deg))
    def test_rotate_L_coords(self, s, incl, az_rot, pa):
        """
        Check that we can rotate automatically to the angular momentum frame.
        """
        assert np.allclose(s.current_rotation, np.eye(3))
        rotmat = L_align(
            s.coordinates_g.get_xyz(),
            s.coordinates_g.differentials["s"].get_d_xyz(),
            s.mHI_g,
            Laxis="x",
        )
        rotmat = rotation_matrix(az_rot, axis="x").T.dot(rotmat)
        rotmat = rotation_matrix(incl, axis="y").T.dot(rotmat)
        if incl >= 0:
            rotmat = rotation_matrix(pa - 90 * U.deg, axis="x").T.dot(rotmat)
        else:
            rotmat = rotation_matrix(pa - 270 * U.deg, axis="x").T.dot(rotmat)
        if U.isclose(pa, 270 * U.deg):
            s.rotate(L_coords=(incl, az_rot))
        else:
            s.rotate(L_coords=(incl, az_rot, pa))
        assert np.allclose(s.current_rotation, rotmat)

    def test_composite_rotations(self, s):
        """
        Check that multiple rotations in a function call are blocked.
        """
        with pytest.raises(
            ValueError, match="Multiple rotations in a single call not allowed."
        ):
            s.rotate(axis_angle=("x", 30 * U.deg), rotmat=np.eye(3))

    def test_translate(self, s):
        """
        Check that coordinates translate correctly.
        """
        for translation_shape in ((3, 1), (1, 3)):
            initial_coords = s.coordinates_g.get_xyz()
            translation = np.ones(3) * U.kpc
            s.translate(translation.reshape(translation_shape))
            expected_coords = initial_coords + translation.reshape((3, 1))
            assert U.allclose(s.coordinates_g.get_xyz(), expected_coords)

    def test_boost(self, s):
        """
        Check that velocities translate correctly.
        """
        for translation_shape in ((3, 1), (1, 3)):
            initial_vels = s.coordinates_g.differentials["s"].get_d_xyz()
            translation = np.ones(3) * U.km / U.s
            s.boost(translation.reshape(translation_shape))
            expected_vels = initial_vels + translation.reshape((3, 1))
            assert U.allclose(
                s.coordinates_g.differentials["s"].get_d_xyz(), expected_vels
            )

    def test_save_current_rotation(self, s):
        """
        Check that current rotation state can be output to file.
        """
        assert np.allclose(s.current_rotation, np.eye(3))
        angle = np.pi / 4
        rotmat = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        s.rotate(rotmat=rotmat)
        testfile = "testrotmat.npy"
        try:
            s.save_current_rotation(testfile)
            saved_rotmat = np.loadtxt(testfile)
        finally:
            if os.path.exists(testfile):
                os.remove(testfile)
        assert np.allclose(saved_rotmat, rotmat)

    @pytest.mark.skipif(
        not have_matplotlib, reason="matplotlib (optional dependency) not available."
    )
    def test_preview(self, s):
        """
        Simply check that the preview visualisation runs without error.
        """
        # with default arguments
        s.preview()
        # with non-default arguments
        s.preview(
            max_points=1000,
            fig=2,
            lim=10 * U.kpc,
            vlim=100 * U.km / U.s,
            point_scaling="fixed",
            title="test",
        )

    @pytest.mark.skipif(
        not have_matplotlib, reason="matplotlib (optional dependency) not available."
    )
    @pytest.mark.parametrize("ext", ("pdf", "png"))
    def test_preview_save(self, s, ext):
        """
        Check that we can output pdf and png preview images.
        """
        testfile = f"preview.{ext}"
        try:
            s.preview(save=testfile)
        finally:
            if os.path.exists(testfile):
                os.remove(testfile)


class TestSOSource:
    @pytest.mark.xfail
    def test_stuff(self):
        raise NotImplementedError


class TestSWIFTGalaxySource:
    @pytest.mark.xfail
    def test_stuff(self):
        raise NotImplementedError


class TestColibreSource:
    @pytest.mark.xfail
    def test_stuff(self):
        raise NotImplementedError


class TestEagleSource:
    @pytest.mark.xfail
    def test_stuff(self):
        raise NotImplementedError


class TestMagneticumSource:
    @pytest.mark.xfail
    def test_stuff(self):
        raise NotImplementedError


class TestSimbaSource:
    @pytest.mark.xfail
    def test_stuff(self):
        raise NotImplementedError


class TestTNGSource:
    @pytest.mark.xfail
    def test_stuff(self):
        raise NotImplementedError
