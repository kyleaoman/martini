from setuptools import setup
import os

with open(
        os.path.join(os.path.dirname(__file__), 'martini', 'VERSION')
) as version_file:
    version = version_file.read().strip()

setup(
    name='astromartini',
    version=version,
    description='Synthetic datacube creation from simulations.',
    url='',
    author='Kyle Oman',
    author_email='koman@astro.rug.nl',
    license='GNU GPL v3',
    packages=['martini'],
    install_requires=['numpy >= 1.15.3', 'astropy >= 3.0', 'scipy'],
    extras_require={
        'hdf5_output': 'h5py',
        'eaglesource': [
            'Hdecompose @ https://github.com/kyleaoman/Hdecompose/'
            'archive/master.zip#egg=Hdecompose',
            'read_eagle @ https://github.com/jchelly/read_eagle/'
            'archive/master.zip#egg=read_eagle'
        ],
        'tngsource': 'Hdecompose @ https://github.com/kyleaoman/Hdecompose/'
        'archive/master.zip#egg=Hdecompose',
        'sosource': [
            'simfiles @ https://github.com/kyleaoman/simfiles/'
            'archive/master.zip#egg=simfiles',
            'simobj @ https://github.com/kyleaoman/simobj/'
            'archive/master.zip#egg=simobj'
        ]
    },
    include_package_data=True,
    zip_safe=False
)
