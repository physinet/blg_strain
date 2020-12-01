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
        'matplotlib==3.3.3',
        'numpy==1.18.1',
        'scipy==1.3.2',
        'h5py==2.10.0',
        'scikit-image==0.17.2'
    ],
    zip_safe=False)
