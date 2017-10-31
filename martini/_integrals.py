import numpy as np

def WendlandC2_line_integral(dx, dy, h):
    retval = np.zeros(h.shape)
    R2 = (dx * dx + dy * dy) / (h * h)
    retval[R2 == 0] = 2. / 3.
    use = np.logical_and(R2 < 1, R2 > 0)
    R2 = R2[use]
    A = np.sqrt(1 - R2)
    retval[use] = 5 * R2 * R2 * (.5 * R2 + 3) * np.log((1 + A) / np.sqrt(R2)) + A * (-27. / 2. * R2 * R2 - 14. / 3. * R2 + 2. / 3.)
    return retval / np.power(h, 2)

#currently line is modelled as a gaussian with fixed width of 7km/s
def Gaussian_integral(a, b, mu=1.0, sigma=1.0):
    return .5 * (erf((b - mu) / (np.sqrt(2.) * sigma)) - erf((a - mu) / (np.sqrt(2.) * sigma)))
