import pathlib
from setuptools import setup
from distutils.core import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='shortcutml',         # How you named your package folder (MyLib)
    packages=['shortcutml'],   # Chose the same as "name"
    version='0.1',      # Start with a small number and increase it with every change you make

    license='MIT',
    # Give a short description about your library
    description='TYPE YOUR DESCRIPTION HERE',
    author='Sulthan Abiyyu Hakim',                   # Type in your name
    author_email='sabiyyuhakim@student.ub.ac.id',      # Type in your E-Mail
    # Provide either the link to your github or to your website
    url='https://github.com/SulthanAbiyyu/ShortcutML',
    # I explain this later on
    download_url='https://github.com/SulthanAbiyyu/ShortcutML/archive/refs/tags/v_01.tar.gz',
    # Keywords that define your package best
    keywords=['machine learning', 'summary'],
    install_requires=[            # I get to this in a second
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
)
