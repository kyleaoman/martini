top_lines = [
    "from setuptools import setup",
    "import os",
    "",
    "with open(",
    "    os.path.join(os.path.dirname(__file__), 'martini', 'VERSION')",
    ") as version_file:",
    "version = version_file.read().strip()",
    "",
    "setup(",
    "    name='astromartini',",
    "    version=version,",
    "    description='Synthetic datacube creation from simulations.',",
    "    url='',",
    "    author='Kyle Oman',",
    "    author_email='koman@astro.rug.nl',",
    "    license='GNU GPL v3',",
    "    packages=['martini'],",
    "    install_requires=['numpy >= 1.15.3', 'astropy >= 3.0', 'scipy'],"
]

er_lines = [
    "    extras_require={",
    "        'hdf5_output': 'h5py',",
    "        'eaglesource': [",
    "            'Hdecompose @ https://github.com/kyleaoman/Hdecompose/'",
    "            'archive/master.zip#egg=Hdecompose',",
    "            'pyread_eagle @ https://github.com/kyleaoman/'",
    "            'pyread_eagle/archive/master.zip#egg=pyread_eagle',",
    "            'eagleSqlTools @ https://github.com/kyleaoman/'",
    "            'eagleSqlTools/archive/master.zip#egg=eagleSqlTools'",
    "        ],",
    "        'tngsource': 'Hdecompose @ https://github.com/kyleaoman/'",
    "        'Hdecompose/archive/master.zip#egg=Hdecompose',",
    "        'sosource': [",
    "            'simfiles @ https://github.com/kyleaoman/simfiles/'",
    "            'archive/master.zip#egg=simfiles',",
    "            'simobj @ https://github.com/kyleaoman/simobj/'",
    "            'archive/master.zip#egg=simobj'",
    "        ],",
    "        'magneticumsource': 'g3t @ https://github.com/kyleaoman/'",
    "        'g3t/archive/master.zip#egg=g3t'",
    "    },"
]

bottom_lines = [
    "    include_package_data=True,",
    "    zip_safe=False",
    ")"
]


def gensetup(for_pypi=False):
    lines = list()
    lines += top_lines
    if not for_pypi:
        lines += er_lines
    else:
        lines += ["    extras_require=dict(),", ]
    lines += bottom_lines
    with open('setup.py', 'w') as f:
        f.writelines([line + '\n' for line in lines])
    return
