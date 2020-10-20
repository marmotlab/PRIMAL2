#!python
#cython: language_level=3

"""
setup
"""

from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension

setup(
    name="astarlib",
    description="A* search algorithm implemented in Cython",
    long_description_content_type="text/markdown",
    url="https://github.com/initbar/astarlib",
    author="Herbert Shin",
    author_email="h@init.bar",
    version="1.0.5",
    license="MIT",
    python_requires=">=3.6",
    install_requires=[
        "Cython>=0.29.12"
    ],
    packages=find_packages(exclude=["docs", "tests"]),
    ext_modules=[Extension("astarlib", sources=["astarlib.pyx"])],
    include_package_data=True,
    zip_safe=False,
    classifiers=[  # https://pypi.org/classifiers
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Utilities",
    ],
)
