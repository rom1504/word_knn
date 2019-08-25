# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='word_knn',
    version='1.0.3',
    description='Quickly find closest words using an efficient knn and word embeddings',
    long_description=readme,
    author='Romain Beaumont',
    author_email='romain.rom1@gmail.com',
    url='https://github.com/rom1504/word_knn',
    license=license,
    packages=find_packages(exclude=('tests')),
    entry_points={
        'console_scripts': [
            'word_knn=word_knn.__main__:main',
        ],
    },
)

