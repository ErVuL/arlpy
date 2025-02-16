from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='arlpy',
    version='1.0.3',
    description='Underwater acoustics toolbox based on arlpy, oalib and pyram',
    long_description=readme,
    author='Theo Bertet',
    author_email='theo.bertet@gmail.com',
    url='https://github.com/ErVuL/arlpy',
    license='BSD (3-clause)',
    keywords='underwater acoustics signal processing communication',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.13.0',
        'utm>=0.7.0',
        'pandas>=1.5.0',
        'bokeh>=3.0.0',
        'matplotlib>=3.9.0'
    ]
)
