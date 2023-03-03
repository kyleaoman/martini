import astropy.units as U
from astropy.coordinates import CartesianRepresentation, CartesianDifferential

def translate(
    cls: CartesianRepresentation, translation_vector: U.Quantity[U.kpc]
) -> CartesianRepresentation: ...
def translate_d(
    cls: CartesianDifferential, translation_vector: U.Quantity[U.km / U.s]
) -> CartesianDifferential: ...
