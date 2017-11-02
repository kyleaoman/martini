import numpy as np
import astropy.units as U

class DataCube():

    def __init__(
            self, 
            n_px_x = 256, 
            n_px_y = 256, 
            n_channels = 64, 
            px_size = 15. * U.arcsec, 
            channel_width = 4. * U.km * U.s ** -1
    ):

        self._array = np.zeros((n_px_x, n_px_y, n_channels))
        self.n_px_x, self.n_px_y, self.n_channels = n_px_x, n_px_y, n_channels
        self.px_size, self.channel_width = px_size, channel_width
        self.pad = 0
            
        return
        
    def spatial_slices(self):
        return iter(self._array.transpose((2, 0, 1)))

    def spectra(self):
        return iter(self._array.reshape(self.n_px_x * self.n_px_y, self.n_channels))

    def add_pad(self, pad):
        tmp = self._array
        self._array = np.zeros((n_px_x + pad * 2, n_px_y + pad * 2, n_channels))
        self._array[self.pad:-self.pad, self.pad:-self.pad, :] = tmp
        self.pad = pad
        return
        
    def drop_pad(self):
        self._array = self._array[self.pad:-self.pad, self.pad:-self.pad, :]
        self.pad = 0
        return

    def __repr__(self):
        return self._array.__repr__()
