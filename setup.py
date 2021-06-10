from distutils.core import setup
setup(
    name = 'InfoPaths',
    packages = ['InfoPaths'],
    version = '0.1',
    license='MIT',
    description = 'TYPE YOUR DESCRIPTION HERE',
    author = 'Benjamin Ahlbrecht',
    author_email = 'BenjaminAhlbrecht@gmail.com',
    url = 'https://github.com/benjamin-ahlbrecht/InfoPaths',
    download_url = 'https://github.com/benjamin-ahlbrecht/InfoPaths/archive/refs/tags/v1.0-alpha.tar.gz',
    keywords = [
        'Information Theory',
        'Transfer Entropy',
        'Mutual Information',
        'Statistical Inference',
        'Conditional Transfer Entropy',
        'Conditional Mutual Information',
        'Nearest Neighbors',
        'Direct Causality',
        'Kraskov, Stoegbauer, Grassberger',
        'Network Analysis',
        'Time Series Analysis',
        'Granger Causality'
    ],
    install_requires=[            # I get to this in a second
        'numpy',
        'sklearn',
        'scipy'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Statisticians',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
