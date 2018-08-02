#!/usr/bin/env python
"""
Generic Catalog Reader (GCR)
A common reader interface for accessing generic catalogs
https://github.com/yymao/generic-catalog-reader
The MIT License (MIT)
Copyright (c) 2017-2018 Yao-Yuan Mao (yymao)
http://opensource.org/licenses/MIT
"""

import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'GCR', 'version.py')) as f:
    exec(f.read()) #pylint: disable=W0122

setup(
    name='GCR',
    version=__version__, #pylint: disable=E0602
    description='Generic Catalog Reader: A common reader interface for accessing generic catalogs',
    url='https://github.com/yymao/generic-catalog-reader',
    download_url='https://github.com/yymao/generic-catalog-reader/archive/v{}.zip'.format(__version__), #pylint: disable=E0602
    author='Yao-Yuan Mao',
    author_email='yymao.astro@gmail.com',
    maintainer='Yao-Yuan Mao',
    maintainer_email='yymao.astro@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='GCR catalog reader',
    packages=['GCR'],
    install_requires=['numpy', 'easyquery>=0.1.3'],
)
