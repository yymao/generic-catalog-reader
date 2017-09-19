#!/usr/bin/env python
"""
Generic Catalog Reader (GCR)
A common reader interface for accessing generic (galaxy/halo) catalogs
The MIT License (MIT)
Copyright (c) 2017 Yao-Yuan Mao (yymao)
http://opensource.org/licenses/MIT
"""

from setuptools import setup

setup(
    name='GCR',
    version='0.1.0',
    description='Generic Catalog Reader: A common reader interface for accessing generic (galaxy/halo) catalogs',
    url='https://github.com/LSSTDESC/generic-catalog-reader',
    author='Yao-Yuan Mao',
    author_email='yymao.astro@gmail.com',
    maintainer='Yao-Yuan Mao',
    maintainer_email='yymao.astro@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='GCR reader',
    packages=['GCR'],
    install_requires=['numpy'],
)
