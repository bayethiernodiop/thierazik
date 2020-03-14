"""Setup for the chocobo package."""

import setuptools

def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]


reqs = parse_requirements("requirements.txt")  

with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Thierno Ibrahima DIOP & Younes RAZIK",
    name='thierazik',
    license="MIT",
    description='A library froked from  jeff heaton (i.e jh-kaggle-util) to fasten the creation of models and ensembling in kaggle or zindi like plateform',
    version='v0',
    long_description=README,
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=reqs,
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
