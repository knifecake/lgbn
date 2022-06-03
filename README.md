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

