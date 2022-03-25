# from setuptools import setup, find_packages
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='cdst',
    packages=setuptools.find_packages(),
    version='1.0',
    author='Dr. Calvin Chan',
    author_email='calvin.chan@bayer.com',
    keywords=['deep learning', 'hyperparameter'],
    url='https://github.com/bayer-int/cdst',
    description="Calvin's Data Science Toolbox",
    long_description=long_description,
    install_requires=['numpy',
                      'scikit-learn',
                      'pandas',
                      'pytorch>="1.9.0"',
                      'skorch',
                      'ray-tune>="1.6.0"',
                      'tensorboardX',
                      'gekko>="1.0.2"',
                      'tensorflow']
)