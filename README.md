# ShortcutML

![banner](./img/banner.png)

## Description

Sometimes, we don’t want to pay attention to detail very much in some machine learning phase. We are in a hurry and only need to prototype our ideas as fast as possible. Then, this library is for you. It’s not making any magic optimization, but it’s summarising all code that maybe you would write when in this kind of situation. This library will be fully customizable from model selection until scoring metrics in the future.

## Installation

```bash
# Clone the repo
$ git clone https://github.com/SulthanAbiyyu/ShortcutML

# Change working dir to shortcutml
$ cd shortcutml

# Install requirements
$ python3 -m pip install -r requirements.txt

# Run one-time setup
$ python3 install.py
```

## Usage

### BaselineModel

```python
from shortcutml.model_selection import BaselineModel

bm = BaselineModel(type="regression") # other type option: "classification"

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
from shortcutml.preprocessing import TextCleaningIndo

tci = TextCleaningIndo()

# Applying all preprocessing process
df["text"] = df["text"].apply(tci.all_preprocessing)
```

### AutoSearchCV

```python
from shortcutml.model_selection import AutoSearchCV

search = AutoSearchCV(model, type="grid") # other type option: "random"
search.search(X,y)

search.cv_results_
```

## Project Plan

- [x] BaselineModel
- [x] TextCleaningIndo
- [ ] TextCleaningEnglish
- [ ] Fully customizable component
- [ ] FeatureSelection -> Pearson, Lasso, Chi Squared, ..
- [x] AutoSearchCV -> Random and Grid
- [ ] ..

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

---

Sulthan Abiyyu, \
16 January 2022
