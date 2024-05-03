# import ez_setup
# ez_setup.use_setuptools()

from setuptools import setup, find_packages


__package__ = 'GRSl2bgen'
__version__ = '1.0.4'

setup(
    name=__package__,
    version=__version__,
    packages=find_packages(exclude=['build']),
    package_data={'': ['*.so', '*h', '*angles*']},
    #     # If any package contains *.txt files, include them:
    #     '': ['*.txt'],
    #     'lut': ['data/lut/*.nc'],
    #     'aux': ['data/aux/*']
    # },
    include_package_data=True,

    url='',
    license='MIT',
    author='Tristan Harmel; Guillaume Morin',
    author_email='tristan.harmel@gmail.com',
    description='scientific code for estimation of the water quality parameters from L2A grs reflectance',

    entry_points={
        'console_scripts': [
            'GRSl2bgen = GRSl2bgen.run:main'
        ]}
)
