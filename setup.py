from setuptools import setup, find_packages

setup(
    name="SpaceGA",  # Project name
    version="1.1.0",  # Initial version
    author="Laust Moesgaard",  # Your name
    author_email="moesgaard@sdu.dk",  # Your email
    description="Package for running SpaceGA on combinatorial small molecule library.",
    long_description=open("README.md").read(),  # Ensure you have a README.md
    long_description_content_type="text/markdown",  # Specify README format
    url="https://github.com/lmoesgaard/SpaceGA",  # GitHub repository URL
    packages=find_packages(),  # Automatically find and include all packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license if different
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Specify the minimum Python version
    install_requires=[
        "beautifulsoup4"
        "h5py"
        "jupyter"
        "numpy"
        "pandas"
        "pyarrow"
        "rdkit-pypi"
        "torch"
    ],
)