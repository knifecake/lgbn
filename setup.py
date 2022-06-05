import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lgbn",
    version="0.1.2",
    author="Elias Hernandis",
    author_email="elias@hernandis.me",
    description="Structure and parameter learning for linear Gaussian Bayesian networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/knifecake/lgbn",
    project_urls={
        "Bug Tracker": "https://github.com/knifecake/lgbn/issues",
        "Documentation": "https://lgbn.readthedocs.io/en/latest/"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=['lgbn', 'lgbn.*']),
    python_requires=">=3.9",
    install_requires=['scipy >= 1.8', 'networkx >= 2.8', 'pandas >= 1.4', 'scikit-learn >= 1.1']
)