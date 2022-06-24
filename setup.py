import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='shortcutml',
    packages=['shortcutml.model_selection',
              'shortcutml.preprocessing',
              'shortcutml.feature_selection'],
    package_dir={
        "shortcutml": "./shortcutml",
    },
    version='0.8',

    license='MIT',
    description='Machine learning baseline prototyping tools',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Sulthan Abiyyu Hakim',
    author_email='sabiyyuhakim@student.ub.ac.id',
    url='https://github.com/SulthanAbiyyu/ShortcutML',

    download_url='https://github.com/SulthanAbiyyu/ShortcutML/archive/refs/tags/0.8.tar.gz',

    keywords=['machine learning', 'summary'],
    install_requires=[
        "joblib",
        "lightgbm",
        "matplotlib",
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
