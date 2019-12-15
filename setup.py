"""Setup for the chocobo package."""

import setuptools


with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Thierno Ibrahima DIOP & Younes RAZIK",
    name='thierazik',
    license="MIT",
    description='A library to fasten the creation of models and ensembling in kaggle like plateform',
    version='v0',
    long_description=README,
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)
