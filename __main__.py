# -*- coding: utf-8 -*-
"""__main__

  Deep Go Shell
"""


import argparse
from deepgo import __version__, __codename__, __release_date__


parser = argparse.ArgumentParser()
parser.add_argument(
    "-V", "--version",
    help="print the Python version number and exit (also --version)\n"
         "when given twice, print more information about the build",
    action="count")


args = parser.parse_args()


if args.version:
  if args.version == 1:
    print(f"Deep Go {__version__}")
  else:
    print(f"Deep Go {__version__} [{__codename__} {__release_date__}]")

