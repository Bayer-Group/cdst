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
    url='https://github.com/Bayer-Group/cdst',
    description="Calvin's Data Science Toolbox",
    long_description=long_description,
    install_requires=['cython>="0.29.24"',
                      'numpy>="1.20.0"',
                      'scikit-learn',
                      'pandas',
                      'torch>="1.9.0"',
                      'skorch',
                      'ray>="1.12.0"',
                      'tensorboardX',
                      'gekko>="1.0.2"',
                      'tensorflow']
)
