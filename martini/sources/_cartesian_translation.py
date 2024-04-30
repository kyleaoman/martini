from astropy.coordinates import CartesianRepresentation, CartesianDifferential

"""
Functions which can be used to extend the
:class:`~astropy.coordinates.representation.cartesian.CartesianRepresentation` and
:class:`~astropy.coordinates.representation.cartesian.CartesianDifferential` classes from
:mod:`astropy.coordinates`. These should be imported, then use:

.. code-block:: python

    # Extend CartesianRepresentation to allow coordinate translation
    setattr(CartesianRepresentation, 'translate', translate)

    # Extend CartesianDifferential to allow velocity (or other differential)
    # translation
    setattr(CartesianDifferential, 'translate', translate_d)

"""

# copied from github.com/kyleaoman/kyleaoman_utilities/kyleaoman_utilities/
# commit-id 81e08768bcf3f910d86757c07b44632f393f29aa
# Note: No git-based solution (e.g. via submodules) seems practical to include
# selected files from external repositories; a direct copy is included here
# to produce a self-contained package.


def translate(cls, translation_vector):
    """
    Apply a coordinate translation.

    Parameters
    ----------
    cls : ~astropy.coordinates.representation.cartesian.CartesianRepresentation
        Equivalent to the ``self`` argument for methods.

    translation_vector : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of length.
        3-vector by which to translate.

    Returns
    -------
    out : ~astropy.coordinates.representation.cartesian.CartesianRepresentation
        A new
        :class:`~astropy.coordinates.representation.cartesian.CartesianRepresentation`
        instance with translation applied.
    """

    return CartesianRepresentation(
        cls.__class__.get_xyz(cls) + translation_vector.reshape(3, 1),
        differentials=cls.differentials,
    )


def translate_d(cls, translation_vector):
    """
    Apply a differential translation.

    Parameters
    ----------
    cls : ~astropy.coordinates.representation.cartesian.CartesianDifferential
        Equivalent to the ``self`` argument for methods.

    translation_vector : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity` with dimensions of velocity (or other
        differential) 3-vector by which to translate.

    Returns
    -------
    out : ~astropy.coordinates.representation.cartesian.CartesianDifferential
        A new
        :class:~astropy.coordinates.representation.cartesian.CartesianDifferential`
        instance with translation applied.
    """

    return CartesianDifferential(
        cls.__class__.get_d_xyz(cls) + translation_vector.reshape(3, 1)
    )
