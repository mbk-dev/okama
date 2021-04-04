from setuptools import setup, find_packages
from pathlib import Path

# read description from README.md and version from okama/__init__.py
(long_description, version) = (None, None)
with (Path(__file__).parent / "README.md").open(encoding="utf-8") as f:
    long_description = f.read()
with (Path(__file__).parent / "okama" / "__init__.py").open(encoding="utf-8") as f:
    for version_line in [l for l in f if l.startswith("__version__") and "=" in l]:
        version = version_line.split("=")[-1].strip().strip("'").strip('"')
if not (long_description and version):
    raise RuntimeError("Unable to get description and version string.")

setup(
    name="okama",
    version=version,
    license="MIT",
    description="Modern Portfolio Theory (MPT) Python package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sergey Kikevich",
    author_email="sergey@rostsber.ru",
    url="https://okama.io/",
    download_url="https://github.com/mbk-dev/okama/archive/v0.81.tar.gz",
    keywords=["finance", "investments", "efficient frontier", "python", "optimization"],
    packages=find_packages(),
    package_data={"tests": ["*.csv"]},
    install_requires=[
        "pandas>=0.25.0",
        "numpy>=1.16.5",
        "scipy>=0.14.0",
        "matplotlib",
        "requests",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
