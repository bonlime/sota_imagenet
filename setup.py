from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().split()

with open("README.md") as f:
    readme = f.read()

setup(
    name="Sota-Imagenet",
    version="0.0.1",
    author_email="bonlimezak@gmail.com",
    packages=find_packages(),
    description="Sota-Imagenet Description",
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    setup_requires=["setuptools_scm"],
    python_requires=">=3, <4",
    license="MIT License",
    install_requires=requirements,
)
