# ShortcutML

![banner](./img/banner.png)

## Description

Sometimes we don’t want to pay attention to detail very much. We are in a hurry and only need to prototype our ideas as fast as possible. Maybe this library is for you. It’s not making any magic optimization or something like that, it’s summarising all code that maybe you would write when in this kind of situation. This library uses scikit-learn, matplotlib, seaborn, pandas, numpy, and NLTK. In the future, this library will be fully customizable. You can choose machine learning models and custom scoring.

## Installation

```bash
# Clone the repo
$ git clone https://github.com/SulthanAbiyyu/ShortcutML

# Change working dir to shortcutml
$ cd shortcutml

# Install requirements
$ python3 -m pip install -r requirements.txt

# Run one-time setup
$ python3 setup.py
```

## Usage

### BaselineModel

```python
from shortcutml import BaselineModel

bm = BaselineModel()

# Evaluate baseline model
bm.evaluate(X_train, X_test, y_train, y_test)

# Plot result
bm.plot_baseline()

# Result dataframe
bm.test_result

# Model lists
bm.classification_models()
bm.regression_models()

# Still not support custom model and scoring
# Default scoring for classification tasks is f1 score and RMSE for regression
```

### TextCleaningIndo

```python
from shortcutml import TextCleaningIndo

tci = TextCleaningIndo()

# Applying all preprocessing process
df["text"] = df["text"].apply(tci.all_preprocessing)
```

## Project Plan

- [x] BaselineModel
- [x] TextCleaningIndo
- [ ] TextCleaningEnglish
- [ ] Fully customizable component
- [ ] FeatureSelection -> Pearson, Lasso, Chi Squared, ..
- [ ] AutoGridSearch
- [ ] ..

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

-----
Sulthan Abiyyu, \
16 January 2022
