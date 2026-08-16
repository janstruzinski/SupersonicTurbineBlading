"""Installation configuration for SupersonicTurbineBlading.

The project intentionally keeps a conventional ``setup.py`` so that the
installation inputs are easy to find for readers familiar with RocketCycles.
``pyproject.toml`` still selects setuptools as the build system and stores the
test configuration.
"""

from setuptools import find_packages, setup

setup(name="supersonic-turbine-blading",
    version="0.1.0",
    description="Sizing of supersonic impulse turbine rotor blades and stator nozzles",
    author="Jan Struzinski",
    license="GPL-3.0-only",
    license_files=["LICENSE"],
    packages=find_packages(include=["SupersonicTurbineBlading", "SupersonicTurbineBlading.*"]),
    python_requires=">=3.10",
    install_requires=["CoolProp>=8.0,<9", "matplotlib>=3.7", "numpy>=1.24", "scipy>=1.10"],
    extras_require={"test": ["pytest>=8"]},
    keywords="supersonic impulse turbine rotor stator nozzle method of characteristics boundary layer",
    classifiers=["Programming Language :: Python :: 3", "Topic :: Scientific/Engineering"],
    zip_safe=False)
