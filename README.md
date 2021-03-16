<p align="center">
    <br>
    <a href="https://github.com/brunneis/ernie#stickers-by-sticker-mule" alt="Stickers section"><img src="misc/ernie-sticker-diecut.png" alt="Ernie Logo" width="150"/></a>
    <br>
<p>

<p align="center">
    <a href="https://pepy.tech/project/ernie/"><img alt="Downloads" src="https://img.shields.io/badge/dynamic/json?style=flat-square&maxAge=3600&label=downloads&query=$.total_downloads&url=https://api.pepy.tech/api/projects/ernie"></a>
    <a href="https://pypi.python.org/pypi/ernie/"><img alt="PyPi" src="https://img.shields.io/pypi/v/ernie.svg?style=flat-square"></a>
    <!--<a href="https://github.com/brunneis/ernie/releases"><img alt="GitHub releases" src="https://img.shields.io/github/release/brunneis/ernie.svg?style=flat-square"></a>-->
    <a href="https://github.com/brunneis/ernie/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/brunneis/ernie.svg?style=flat-square"></a>
</p>

<h3 align="center">
    <b>BERT's best friend.</b>
</h3>

<p align="center">
    <a href="https://www.buymeacoffee.com/brunneis" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35px"></a>
</p>

# Installation
> Ernie requires Python 3.6 or higher.
```bash
pip install ernie
```
<a href="https://colab.research.google.com/drive/10lmqZyAHFP_-x4LxIQxZCavYpPqcR28c"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg?style=flat-square"></a>

# Fine-Tuning
## Sentence Classification
```python
from ernie import SentenceClassifier, Models
import pandas as pd

tuples = [("This is a positive example. I'm very happy today.", 1),
          ("This is a negative sentence. Everything was wrong today at work.", 0)]

df = pd.DataFrame(tuples)
classifier = SentenceClassifier(model_name=Models.BertBaseUncased, max_length=64, labels_no=2)
classifier.load_dataset(df, validation_split=0.2)
classifier.fine_tune(epochs=4, learning_rate=2e-5, training_batch_size=32, validation_batch_size=64)
```



