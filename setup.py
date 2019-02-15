from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = [
    "dotmap",
    "numpy",
    "tqdm>=4.24.0",
    "scipy>=1.1.0",
    "keras>=2.2.4"
  ]

setup(
  name="ultrasound-research",
  version="0.1",
  package_dir={"":"src"},
  packages=find_packages("src"),
  description="Ultrasound research repository",
  author="Matthew Goodman",
  author_email="mattgoodman13@gmail.com",
  license="BSD3",
  include_package_data=True,
  install_requires=REQUIRED_PACKAGES)