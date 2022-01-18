import pathlib
from setuptools import setup, find_packages
from distutils.core import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='shortcutml',
    packages=['shortcutml'],
    version='0.3',

    license='MIT',
    # Give a short description about your library
    description='Machine learning baseline prototyping tools',
    long_description=README,
    long_description_content_type="text/markdown",
    author='Sulthan Abiyyu Hakim',
    author_email='sabiyyuhakim@student.ub.ac.id',
    url='https://github.com/SulthanAbiyyu/ShortcutML',

    download_url='https://github.com/SulthanAbiyyu/ShortcutML/archive/refs/tags/0.3.tar.gz',

    keywords=['machine learning', 'summary'],
    install_requires=[
        "joblib",
        "lightgbm",
        "matplotlib",
        "nltk",
        "numpy",
        "pandas",
        "PySastrawi",
        "scikit-learn",
        "seaborn",
        "xgboost"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
)
