import pytest
import numpy as np
from astropy import units as U
from martini.sources import SPHSource
from astropy.units import isclose, allclose
from martini.sources._cartesian_translation import translate, translate_d
from astropy.coordinates import CartesianRepresentation, CartesianDifferential


class TestSourceUtilities:
    @pytest.mark.xfail
    def test_L_align(self):
        raise NotImplementedError

    def test_translate(self):
        """
        Check that cartesian representation transforms correctly.
        """
        setattr(CartesianRepresentation, "translate", translate)
        cr = CartesianRepresentation(np.zeros(3) * U.kpc)
        translation = np.ones(3) * U.kpc
        cr_translated = cr.translate(translation)
        assert allclose(cr_translated.get_xyz(), translation)

    def test_translate_d(self):
        """
        Check that cartesian differential transforms correctly.
        """
        setattr(CartesianDifferential, "translate", translate_d)
        cd = CartesianDifferential(np.zeros(3) * U.km / U.s)
        translation = np.ones(3) * U.km / U.s
        cd_translated = cd.translate(translation)
        assert allclose(cd_translated.get_d_xyz(), translation)


class TestSPHSource:
    def test_coordinate_input(self):
        """
        Check that different input shapes for coordinates give expected behaviour.
        """
        row_coords = np.zeros((4, 3)) * U.kpc
        row_vels = np.zeros((4, 3)) * U.km / U.s
        s = SPHSource(xyz_g=row_coords, vxyz_g=row_vels)
        assert s.coordinates_g.shape == (4,)
        assert s.coordinates_g.differentials["s"].shape == (4,)
        assert s.npart == 4
        col_coords = np.zeros((3, 4)) * U.kpc
        col_vels = np.zeros((3, 4)) * U.km / U.s
        s = SPHSource(xyz_g=col_coords, vxyz_g=col_vels)
        assert s.coordinates_g.shape == (4,)
        assert s.coordinates_g.differentials["s"].shape == (4,)
        assert s.npart == 4
        symm_coords = np.zeros((3, 3)) * U.kpc
        symm_coords[0, 1] = 1 * U.kpc
        symm_vels = np.zeros((3, 3)) * U.km / U.s
        symm_vels[0, 1] = 1 * U.km / U.s
        s = SPHSource(xyz_g=symm_coords, vxyz_g=symm_vels, coordinate_axis=0)
        assert s.coordinates_g.shape == (3,)
        assert s.coordinates_g.differentials["s"].shape == (3,)
        assert s.coordinates_g.x[1] == 1 * U.kpc + s.distance
        assert s.coordinates_g.differentials["s"].d_x[1] == 1 * U.km / U.s + s.vsys
        assert s.npart == 3
        s = SPHSource(xyz_g=symm_coords, vxyz_g=symm_vels, coordinate_axis=1)
        assert s.coordinates_g.shape == (3,)
        assert s.coordinates_g.shape == (3,)
        assert s.coordinates_g.y[0] == 1 * U.kpc
        assert s.coordinates_g.differentials["s"].d_y[0] == 1 * U.km / U.s
        assert s.npart == 3
        with pytest.raises(RuntimeError, match="cannot guess coordinate_axis"):
            SPHSource(xyz_g=symm_coords, vxyz_g=symm_vels)
        with pytest.raises(ValueError, match="must have matching shapes"):
            SPHSource(xyz_g=row_coords, vxyz_g=col_vels)

    @pytest.mark.parametrize("ra", (0 * U.deg, 30 * U.deg, -30 * U.deg))
    @pytest.mark.parametrize("dec", (0 * U.deg, 30 * U.deg, -30 * U.deg))
    def test_ra_dec_rotation(self, ra, dec):
        """
        Check that coordinates rotate as needed before translation to observed position.
        """
        xyz_g = np.arange(3).reshape((1, 3)) * U.kpc
        vxyz_g = np.arange(3).reshape((1, 3)) * U.km / U.s
        s = SPHSource(xyz_g=xyz_g, vxyz_g=vxyz_g, ra=ra, dec=dec, distance=0 * U.Mpc)
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
        assert allclose(s.coordinates_g.xyz.T, xyz_g.dot(rotmat))
        assert allclose(s.coordinates_g.differentials["s"].d_xyz.T, vxyz_g.dot(rotmat))

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
        s = SPHSource(
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            ra=ra,
            dec=dec,
            distance=distance,
            vpeculiar=vpeculiar,
        )
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
        assert allclose(
            s.coordinates_g.xyz.T, xyz_g.dot(rotmat) + direction_vector.T * distance
        )
        assert allclose(
            s.coordinates_g.differentials["s"].d_xyz.T,
            vxyz_g.dot(rotmat) + direction_vector.T * vsys,
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
        s = SPHSource(xyz_g=xyz_g, vxyz_g=vxyz_g, ra=ra, dec=dec)
        print(s.sky_coordinates.ra[0], Angle(ra).wrap_at(360 * U.deg))
        assert isclose(s.sky_coordinates.ra[0], Angle(ra).wrap_at(360 * U.deg))
        assert isclose(s.sky_coordinates.dec[0], Angle(dec).wrap_at(180 * U.deg))

    @pytest.mark.xfail
    def test_apply_mask(self, s):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_rotate_axis_angle(self, s):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_rotate_rotmat(self, s):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_rotate_L_coords(self, s):
        raise NotImplementedError

    def test_translate_position(self, s):
        """
        Check that coordinates translate correctly.
        """
        for translation_shape in ((3, 1), (1, 3)):
            initial_coords = s.coordinates_g.get_xyz()
            translation = np.ones(3) * U.kpc
            s.translate_position(translation.reshape(translation_shape))
            expected_coords = initial_coords + translation.reshape((3, 1))
            assert allclose(s.coordinates_g.get_xyz(), expected_coords)

    def test_translate_velocity(self, s):
        """
        Check that velocities translate correctly.
        """
        for translation_shape in ((3, 1), (1, 3)):
            initial_vels = s.coordinates_g.differentials["s"].get_d_xyz()
            translation = np.ones(3) * U.km / U.s
            s.translate_velocity(translation.reshape(translation_shape))
            expected_vels = initial_vels + translation.reshape((3, 1))
            assert allclose(
                s.coordinates_g.differentials["s"].get_d_xyz(), expected_vels
            )

    @pytest.mark.xfail
    def test_save_current_rotation(self, s):
        raise NotImplementedError


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
