from setuptools import setup
import os

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as version_file:
    version = version_file.read().strip()

setup(
    name='astromartini',
    version=version,
    description='Synthetic datacube creation from simulations.',
    url='',
    author='Kyle Oman',
    author_email='koman@astro.rug.nl',
    license='',
    packages=['martini'],
    install_requires=['numpy', 'astropy', 'scipy'],
    include_package_data=True,
    zip_safe=False
)
