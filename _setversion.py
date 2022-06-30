import os
from sys import argv
from _gencodemeta import gencodemeta

version = argv[1]

with open(
    os.path.join(os.path.dirname(__file__), "martini", "VERSION"), "w"
) as version_file:
    version_file.write(version)

gencodemeta()
