from setuptools import setup

setup(name='blg_strain',
    version='0.1',
    description='Band structure calculations for strained bilayer graphene',
    url='http://github.com/physinet/blg_strain',
    author='Brian Schaefer',
    author_email='physinet@gmail.com',
    license='MIT',
    packages=['blg_strain'],
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy'
    ],
    zip_safe=False)
