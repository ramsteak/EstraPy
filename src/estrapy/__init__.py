from importlib.metadata import version
from packaging.version import Version


__version__ = version("estrapy")
__version_tuple__ = Version(__version__).release

author = 'Marco Stecca'
copyright = f'(c) 2024-2026 {author}'

banner = fr"""
    ______     __             ____       
   / ____/____/ /__________ _/ __ \__  __
  / __/ / ___/ __/ ___/ __ `/ /_/ / / / /
 / /___(__  ) /_/ /  / /_/ / ____/ /_/ /
/_____/____/\__/_/   \__,_/_/    \__, /
                                /____/
EstraPy - Data Analysis Framework
Version: {__version__}
Copyright: {copyright}""".removeprefix("\n")
