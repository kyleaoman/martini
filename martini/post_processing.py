from astropy import units


def get_mass_from_flux(flux, beam, pixelSize, channelSize, distance):
    beamArea = (beam.sr / (pixelSize**2)).decompose()
    return (
        2.356e5
        * (distance / units.Mpc).decompose() ** 2
        * (flux / units.Jy).decompose()
        / beamArea
        * (channelSize / (units.km / units.s)).decompose()
        * units.Msun
    )


def get_Jy_from_mass(cube, beam, pixelSize, channelWidth, distance):
    converter = get_mass_from_flux(
        1 * units.Jy, beam, pixelSize, channelWidth, distance
    )
    return (cube / converter).decompose() * units.Jy / units.beam
