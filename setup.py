from setuptools import setup, find_packages

setup(
    name="Soundscapy",
    version="0.3.1",
    packages=find_packages(exclude=["*test", "examples"]),
)