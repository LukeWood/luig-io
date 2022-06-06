from setuptools import find_packages, setup

VERSION = "0.0.0"

setup(
    name="luig-io",
    packages=find_packages(exclude=("*_test.py",)),
    version=VERSION,
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/lukewood/luig-io",
    author="Luke Wood",
    author_email="lukewoodcs@gmail.com",
    install_requires=[
        "black",
        "isort",
        "flake8",
        "tensorflow",
        "gym-super-mario-bros",
        "gym==0.22.0",
    ],
)
