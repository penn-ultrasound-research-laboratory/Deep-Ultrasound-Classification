from setuptools import setup
from setuptools import find_packages

setup(
  name="ultrasound-research",
  version="0.1",
  packages=find_packages("src"),
  description="Ultrasound research repository",
  author="Matthew Goodman",
  author_email="mattgoodman13@gmail.com",
  license="BSD3",
  install_requires=[],
  zip_safe=False)