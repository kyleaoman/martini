"""
Provide extensions enabling coordinate translations on :mod:`astropy` coordinate objects.

Functions which can be used to extend the
:class:`~astropy.coordinates.representation.cartesian.CartesianRepresentation` and
:class:`~astropy.coordinates.representation.cartesian.CartesianDifferential` classes from
:mod:`astropy.coordinates`.

These should be imported, then use:

.. code-block:: python

    # Extend CartesianRepresentation to allow coordinate translation
    setattr(CartesianRepresentation, 'translate', translate)

    # Extend CartesianDifferential to allow velocity (or other differential)
    # translation
    setattr(CartesianDifferential, 'translate', translate_d)
"""

# Adapted from github.com/kyleaoman/kyleaoman_utilities/kyleaoman_utilities/
# commit-id 81e08768bcf3f910d86757c07b44632f393f29aa
# Note: No git-based solution (e.g. via submodules) seems practical to include
# selected files from external repositories; a direct copy is included here
# to produce a self-contained package.

import numpy as np
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
import astropy.units as U


def _apply_affine_transform(
    coords: U.Quantity, affine_transform: np.ndarray, transform_units: U.UnitBase
) -> U.Quantity:
    """
    Apply an affine coordinate transformation to a coordinate array.

    An arbitrary coordinate transformation mixing translations and rotations can be
    expressed as a 4x4 matrix. However, such a matrix has mixed units, so we need to
    assume a consistent unit for all transformations and work with bare arrays.

    Parameters
    ----------
    coords : ~astropy.units.Quantity
        The coordinate array to be transformed.

    transform : ~numpy.ndarray
        The 4x4 affine transformation matrix.

    transform_units : ~astropy.units.UnitBase
        The units assumed in the translation portion of the transformation matrix.

    Returns
    -------
    ~astropy.units.Quantity
        The coordinate array with transformation applied.
    """
    return (
        U.Quantity(
            np.dot(
                affine_transform,
                np.vstack(
                    (
                        coords.to_value(transform_units),
                        np.ones(coords.shape[1])[np.newaxis],
                    )
                ),
            )[:3, :],
            transform_units,
        )
        << coords.unit
    )


def translate(
    cls: CartesianRepresentation, translation_vector: U.Quantity[U.kpc]
) -> CartesianRepresentation:
    """
    Apply a coordinate translation.

    Parameters
    ----------
    cls : ~astropy.coordinates.representation.cartesian.CartesianRepresentation
        The cartesian representation instance.

    translation_vector : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of length.
        3-vector by which to translate.

    Returns
    -------
    ~astropy.coordinates.representation.cartesian.CartesianRepresentation
        A new
        :class:`~astropy.coordinates.representation.cartesian.CartesianRepresentation`
        instance with translation applied.
    """
    return CartesianRepresentation(
        cls.__class__.get_xyz(cls) + translation_vector.reshape(3, 1),
        differentials=cls.differentials,
    )


def translate_d(
    cls: CartesianDifferential, translation_vector: U.Quantity[U.km / U.s]
) -> CartesianDifferential:
    """
    Apply a differential translation.

    Parameters
    ----------
    cls : ~astropy.coordinates.representation.cartesian.CartesianDifferential
        The cartesian differential instance.

    translation_vector : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity` with dimensions of velocity (or other
        differential) 3-vector by which to translate.

    Returns
    -------
    ~astropy.coordinates.representation.cartesian.CartesianDifferential
        A new
        :class:~astropy.coordinates.representation.cartesian.CartesianDifferential`
        instance with translation applied.
    """
    return CartesianDifferential(
        cls.__class__.get_d_xyz(cls) + translation_vector.reshape(3, 1)
    )


def affine_transform(
    cls: CartesianRepresentation,
    affine_transform: np.ndarray,
    transform_units: U.UnitBase,
) -> CartesianRepresentation:
    """
    Apply an affine coordinate transformation to a coordinate array.

    An arbitrary coordinate transformation mixing translations and rotations can be
    expressed as a 4x4 matrix. However, such a matrix has mixed units, so we need to
    assume a consistent unit for all transformations and work with bare arrays.

    Parameters
    ----------
    cls : ~astropy.coordinates.representation.cartesian.CartesianRepresentation
        The cartesian representation instance.

    affine_transform : ~numpy.ndarray
        The 4x4 affine transformation matrix.

    transform_units : ~astropy.units.UnitBase
        The units assumed for the translation portion of the matrix.

    Returns
    -------
    ~astropy.coordinates.representation.cartesian.CartesianRepresentation
        A new
        :class:`~astropy.coordinates.representation.cartesian.CartesianRepresentation`
        instance with transformation applied.
    """
    return CartesianRepresentation(
        _apply_affine_transform(
            cls.__class__.get_xyz(cls),
            affine_transform,
            transform_units=transform_units,
        ),
        differentials=cls.differentials,
    )


def affine_transform_d(
    cls: CartesianDifferential,
    affine_transform: np.ndarray,
    transform_units: U.UnitBase,
) -> CartesianDifferential:
    """
    Apply a differential affine transformation.

    An arbitrary coordinate transformation mixing translations and rotations can be
    expressed as a 4x4 matrix. However, such a matrix has mixed units, so we need to
    assume a consistent unit for all transformations and work with bare arrays.

    Parameters
    ----------
    cls : ~astropy.coordinates.representation.cartesian.CartesianDifferential
        The cartesian differential instance.

    affine_transform : ~numpy.ndarray
        The 4x4 affine transformation matrix.

    transform_units : ~astropy.units.UnitBase
        The units assumed for the translation portion of the matrix.

    Returns
    -------
    ~astropy.coordinates.representation.cartesian.CartesianDifferential
        A new
        :class:~astropy.coordinates.representation.cartesian.CartesianDifferential`
        instance with translation applied.
    """
    return CartesianDifferential(
        _apply_affine_transform(
            cls.__class__.get_d_xyz(cls),
            affine_transform,
            transform_units=transform_units,
        )
    )
