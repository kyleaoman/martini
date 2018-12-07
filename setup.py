from setuptools import setup

setup(
    name='astromartini',
    version='1.0.0',
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
