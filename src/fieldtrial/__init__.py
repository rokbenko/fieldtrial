"""fieldtrial: statistically rigorous real-world evaluation of robot policies.

Find out whether your robot policy actually got better.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fieldtrial")
except PackageNotFoundError:  # pragma: no cover - imported from an uninstalled source tree
    __version__ = "0+unknown"

__all__ = ["__version__"]
