import os


class Lines(list):
    ind = "    "

    def __init__(self, *args, **kwargs):
        self.nind = 0
        super().__init__(*args, **kwargs)
        return

    def indent(self):
        self.nind += 1
        return

    def unindent(self):
        self.nind -= 1
        return

    def append(self, l):
        super().append(self.nind * self.ind + l + "\n")
        return


def gencodemeta():
    with open(
        os.path.join(os.path.dirname(__file__), "martini", "VERSION")
    ) as version_file:
        version = version_file.read().strip()

    fields = {
        "@context": "https://doi.org/10.5063/schema/codemeta-2.0",
        "@type": "SoftwareSourceCode",
        "name": "MARTINI: Mock spatially resolved spectral line observations "
        "of simulated galaxies",
        "description": "MARTINI (Mock Array Radio Telescope Interferometry of "
        "the Neutal ISM) creates synthetic resolved HI line observations (data"
        " cubes) of smoothed-particle hydrodynamics simulations of galaxies. "
        "The various aspects of the mock-observing process are divided "
        "logically into sub-modules handling the data cube, source, beam, "
        "noise, spectral model and SPH kernel. MARTINI is object-oriented: "
        "each sub-module provides a class (or classes) which can be configured"
        " as desired. For most sub-modules, base classes are provided to allow"
        " for straightforward customization. Instances of each sub-module "
        "class are given as parameters to the Martini class; a mock "
        "observation is then constructed by calling a handful of functions to"
        " execute the desired steps in the mock-observing process.",
        "identifier": "ascl:1911.005",
        "author": [
            {
                "@type": "Person",
                "givenName": "Kyle A.",
                "familyName": "Oman",
                "@id": "0000-0001-9857-7788",
            }
        ],
        "citation": "https://ui.adsabs.harvard.edu/abs/2019ascl.soft11005O",
        "relatedLink": ["https://pypi.org/project/astromartini"],
        "codeRepository": ["https://github.com/kyleaoman/martini"],
        "version": version,
        "license": "https://spdx.org/licenses/GPL-3.0-only.html",
    }

    L = Lines()

    L.append("{")
    L.indent()
    for k, v in fields.items():
        if isinstance(v, str):
            L.append('"{:s}": "{:s}",'.format(k, v))
        elif isinstance(v, list):
            L.append('"{:s}": ['.format(k))
            L.indent()
            for l in v:
                if isinstance(l, str):
                    L.append('"{:s}",'.format(l))
                elif isinstance(l, dict):
                    L.append("{")
                    L.indent()
                    for kk, vv in l.items():
                        L.append('"{:s}": "{:s}",'.format(kk, vv))
                    L.unindent()
                    L.append("},")
                else:
                    raise RuntimeError("Unhandled!")
            L.unindent()
            L.append("],")
        else:
            raise RuntimeError("Unhandled!")
    L.unindent()
    L.append("}")

    with open("codemeta.json", "w") as f:
        f.writelines(L)
