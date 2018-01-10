from setuptools import setup

setup(
    name='martini',
    version='0.1',
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
