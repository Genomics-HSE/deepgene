from setuptools import setup, find_packages

setup(
    name='deepgene',
    version='0.1',
    license='MIT',
    packages=find_packages(where='deepgene/', ),
    package_dir={'': 'deepgene/'},
)
