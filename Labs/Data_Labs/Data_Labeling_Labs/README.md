# The Significance of Data Labeling and the Role of Snorkel in Enhancing Machine Learning

In the realm of machine learning (ML) and artificial intelligence (AI), data is the bedrock upon which models are built and refined. However, raw data, while abundant, often lacks the structure and organization necessary for effective learning. This is where data labeling comes into play. Data labeling is the process of annotating or tagging data points with relevant metadata or labels, providing context and meaning to otherwise unstructured information. This process is crucial for numerous reasons, and its importance cannot be overstated in the development and deployment of ML models.

- First and foremost, labeled data serves as the training material for ML algorithms. Supervised learning, one of the most prevalent paradigms in ML, relies heavily on labeled datasets to learn the relationship between input features and target outputs. Without accurately labeled data, ML models would struggle to generalize patterns and make accurate predictions. Thus, the quality and accuracy of labels directly influence the performance and reliability of ML systems.

- Furthermore, data labeling facilitates the creation of ground truth datasets, which serve as benchmarks for evaluating model performance. By comparing model predictions against accurately labeled data, ML engineers can assess the efficacy of their algorithms and identify areas for improvement. This iterative feedback loop is essential for refining ML models and enhancing their robustness in real-world scenarios.

- Additionally, labeled data enables domain-specific insights and knowledge extraction. By categorizing data points into meaningful classes or attributes, organizations can derive valuable insights about customer preferences, market trends, and business operations. These insights can inform strategic decision-making and drive innovation across various industries, from healthcare to finance to e-commerce.

Despite its undeniable importance, data labeling can be a labor-intensive and time-consuming process, especially when dealing with large volumes of unstructured data. Manual labeling by human annotators is often prone to errors, inconsistencies, and biases, leading to suboptimal model performance. This is where automated labeling tools like Snorkel come into play.

`Snorkel is a powerful framework designed to streamline the data labeling pipeline and mitigate the challenges associated with manual annotation. By leveraging weak supervision techniques, Snorkel enables ML teams to programmatically generate labels from noisy or imperfect sources, such as heuristics, rules, or distant supervision. This approach not only accelerates the labeling process but also improves label quality by aggregating information from multiple sources and learning from noisy signals.`

Moreover, Snorkel provides a unified platform for managing the entire data labeling workflow, from data ingestion to model training. Its flexible and extensible architecture allows users to customize labeling functions, integrate with existing ML pipelines, and adapt to evolving data requirements. By automating tedious labeling tasks and empowering ML engineers to focus on high-level model design and optimization, Snorkel enhances productivity and accelerates the pace of innovation in ML research and development.

In this toturail we will learn fundamental aspects of Snorkel using 3 examples provided by Snorkel team.

# Spam Tutorials
We consider a canonical machine learning problem: classifying spam. This directory contains three tutorials, described below:
* `01_spam_tutorial`: This tutorial dives deep into how we can create, analyze, and use labeling functions for the spam classification task.
* `02_spam_data_augmentation_tutorial`: This tutorial demonstrates how to write, combine and apply transformation functions for performing data augmentation.
* `03_spam_data_slicing_tutorial`: This tutorial shows how we can use slicing functions to identify important slices of the data, for monitoring and improved performance.

---

# Changes Made in This Submission

The three original tutorial notebooks are kept as baseline runs. All changes are in separate `*_experiments.ipynb` copies: two experiments per notebook, each with a hypothesis, the code change, a results table and an explanation in a Markdown cell right after it.

| File | Status |
|---|---|
| `01_spam_tutorial.ipynb` | Original, re-run as the baseline |
| `01_spam_lf_experiments.ipynb` | **New**: labeling function experiments |
| `02_spam_data_augmentation_tutorial.ipynb` | Original + Keras 3 compatibility fix (see below), re-run as the baseline |
| `02_spam_data_augmentation_experiments.ipynb` | **New**: augmentation experiments (in an "Experiments" section at the end) |
| `03_spam_data_slicing_tutorial.ipynb` | Original, re-run as the baseline |
| `03_spam_slicing_experiments.ipynb` | **New**: slicing experiments (in an "Experiments" section before the Recap) |
| `BASELINE_RESULTS.md` | **New**: baseline metrics and LF summary table for notebook 01 |
| `utils.py`, `requirements.txt`, `data/` | Unchanged |

## Environment

- Python 3.11 virtual environment inside this folder (`.venv/`, gitignored), with `pip install -r requirements.txt`.
- Extra downloads not covered by `requirements.txt`: `python -m spacy download en_core_web_sm`, `python -m textblob.download_corpora`, and NLTK `wordnet`.
- Resolved versions included snorkel 0.10.0, TensorFlow 2.21 (Keras 3), PyTorch 2.15 and scikit-learn 1.9.
- **Compatibility fix (notebook 02):** the "make Keras reproducible" cell used `tf.compat.v1.Session` / `set_session`, which Keras 3 removed. It was replaced with `tf.keras.utils.set_random_seed(0)`. Seeds are still fixed, but execution is no longer forced to one thread, so LSTM results can vary slightly between runs.

## Notebook 01: Labeling Functions

Baseline (original 10 LFs): Majority Vote **83.6%**, Label Model **86.4%**, classifier test accuracy **93.2%**.
The data loader now also returns the labeled dev split (`split_dev_valid=True`); the train and test sets are unchanged.

| # | Change | Label Model | Test acc. |
|---|---|---|---|
| 1 | `keyword_my` → `keyword_my_channel` ("my channel", "my video", "my music", "my page") | 87.2% | 92.4% |
| 2 | + `textblob_subjectivity_strict` (threshold 0.5 → 0.8), cumulative with #1 | 88.0% | 84.4% |

**Findings:** The bare keyword "my" matches substrings ("Amy", "economy"), so tightening it halved its conflicts. The effect on the results was within noise (±2 test comments). The stricter subjectivity LF produced the most accurate labels so far, but the classifier dropped about 9 points. That LF was a main source of HAM votes, and comments with no votes are filtered out before training, so the classifier saw fewer genuine comments. Better labels with less coverage gave a worse model.

## Notebook 02: Data Augmentation

Baseline: 47.2% (original data) vs 71.6% (augmented data).

| # | Change | Result |
|---|---|---|
| 1 | Train for 20 epochs instead of 5, and report the share of SPAM predictions | Original data: still 47.2%, predicting SPAM for 100% of comments. Augmented: 72.0% |
| 2 | New TF `delete_random_word`, added to the mean-field policy (20 epochs) | Augmented set grew to 3,323 comments; the model collapsed to predicting SPAM for 100% of comments (47.2%) |

**Findings:** The test set is 118 SPAM / 132 HAM, so 47.2% means the baseline model predicts SPAM for every comment. The tutorial's "augmentation improved the model" compares against a model that never learned. The original tutorial outputs show the same numbers, so this isn't caused by our environment. The results point to the LSTM setup (Adagrad optimizer with its small default learning rate, and a ReLU activation in the LSTM) as the problem, rather than the data.

## Notebook 03: Data Slicing

Baseline: logistic regression F1 **92.5%**; slice-aware model F1 **93.2%**. All slice tables now also show slice size, spam share and accuracy.

| # | Change | Result |
|---|---|---|
| 1 | New SFs `has_url` (http / www. / .com) and `mostly_caps` (>50% uppercase letters, at least 10 letters), monitoring only | Neither is a weak spot: `has_url` 16/16 correct, `mostly_caps` 17/18 correct |
| 2 | Slice-aware model trained with the 2 new SFs vs the tutorial's 5 (same seed and settings) | Overall accuracy 94.0% → 91.6%; `has_url` 100% → 93.8% |

**Findings:** The tutorial's weakest-looking slice, `short_comment` (66.7% F1), actually has 94.6% accuracy. Only 9% of it is spam, and F1 only scores the SPAM class. `short_link` has just 3 test comments. Adding slice heads where the model wasn't weak made it worse: the extra heads were under-trained in 2 epochs, and the slices are tiny. Slices are most useful for monitoring, and slice-aware training should target slices that are actually weak.

**Caveat (all notebooks):** the test set has 250 comments and each result comes from a single run, so differences of 1–2% (3–5 comments) are within noise.
