# lgbn

Structure learning for linear Gaussian Bayesian networks.

## Development

Development has been done on a Linux machine using Python 3.9. Although we strived to make the code portable, the package might not work fully on other platforms. This section covers how to get up and running contributing to this package on a Linux-compatible setup (MacOS and WSL should mostly work).

### Development setup

Clone the repository and run the following commands inside the repository root:

    python3 -m pip install virtualenv
    python3 -m virtualenv .venv
    source .venv/bin/activate

    pip install -r requirements.txt

These commands will create a virtual environment and install all dependencies isolated from the rest of your system. Verify everything is working by running the test suite (see next section).

### Testing

This repository includes unit tests for the Python package written using the `unittest` module of the Python standard library. To run the tests we rely on `unittest`'s autodiscovery feature:

    python3 -m unittest

### Building the package

Ensure you have installed all dev dependencies. Then run

    python3 setup.py sdist

from the repository root. This will build the package and place it under `dist/lgbn-x.y.z.tar.gz`. You can install it on your own machine by running

    python3 -m pip install --user dist/lgbn-x.y.z.tar.gz


### Building documentation

Documentation is writen in [reStructuredText](https://docutils.sourceforge.io/docs/user/rst/quickref.html "A quick reference on reStructuredText") and consists of two parts:

 - API reference documentation is written in the *docstrings* of classes, methods, etc. following the [numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html). These are then imported into the documentation using autodoc.

 - Usage and design documentation, consisting of long form documents which explain how to work with the package and the design decisions behind it. These documents are stored in the `docs/` directory.

Documentation can be compiled in to a set of HTML documents plus many other formats by using Sphinx. To build the HTML version documentation use

    cd docs
    make html


## Acknowledgements

This package was developed while writing the bachelor's thesis of the author at Universidad Autónoma de Madrid. The Bachelor's thesis was supervised by Daniel Ramos with help from Pablo Ramírez Hereza, both members of the Audio, Data Intelligence and Speech research group (AUDIAS) at Escuela Politécnica Superior (UAM).

Motivation came from the lack of support for linear Gaussian Bayesian networks in other Python packages like pgmpy, which has served as a source of inspiration for the structure of the package.

## License

This package and its documentation are copyright of Elias Hernandis. They are released under the MIT license. The MIT License grants anyone permission to use, copy, modify and distribute versions of this software without limitations, including for comercial purposes. This software is provided without any warranties. Please see the LICENSE file for more information.