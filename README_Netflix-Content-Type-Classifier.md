# 🎬 Netflix Content Type Classifier

An end-to-end **machine learning classification project** that predicts whether a Netflix title is a **🎬 Movie** or a **📺 TV Show**, built on the public Netflix Titles dataset (**8,807 titles, 12 columns**).

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-boosting-189ABD)

---

## 🧭 Workflow

```
Load data → EDA → Cleaning → Missing values → Duplicates
   → Feature engineering → Encoding → Correlation analysis
   → Feature selection (RFE) → Train/test split → Scaling → PCA
   → Train 6 models → Evaluate → 5-fold cross-validation
   → Pick best model → Save model + preprocessing → Reload and predict
```

---

## 📂 Files

| File | Purpose |
|------|---------|
| `Netflix Content Type Classifier.ipynb` | 🧠 The main notebook: the full pipeline above, from raw data to saved model. |
| `02_load_model_and_predict.ipynb` | 🔁 Loads the saved model and preprocessing objects and predicts on random rows, without retraining. |
| `test_on_best_model.ipynb` | 🧪 A small test notebook that loads the saved best model and checks its predictions. |
| `netflix_titles_CLEANED.csv` | 🗃️ The Netflix titles dataset used by the project. |

Running the main notebook generates the reusable artifacts: `best_model.pkl`, `label_type.pkl`, `feature_cols.pkl`, `scaler.pkl`, `netflix_processed.csv` (and `best_model_ann.keras` when the neural network is the best model). The two other notebooks expect these files in the same folder.

---

## 🧹 Data preparation

- 🕳️ **Missing values:** `directors`, `cast` and `countries` filled with `"Unknown"`; `rating` and `duration` with the mode; `date_added` with forward fill.
- 🛠️ **Data-quality fix:** some `rating` cells actually contained durations (for example `74 min`). They are detected with a mask, moved to `duration`, and the rating is set to missing.
- 🗑️ **Duplicates** removed.
- ⚙️ **Feature engineering:** `num_cast`, `num_directors`, `num_countries`, `num_genres`, `main_country`, `main_genre`.
- 🔎 **Correlation analysis** (threshold 0.9) and **Recursive Feature Elimination** keeping 5 features: `num_directors`, `num_genres`, `rating`, `main_country`, `main_genre`.
- 📏 **StandardScaler** fitted on the training set only, to avoid leakage from the test set.
- 📉 **PCA** (5 components) explaining about **82.5%** of the variance.
- ✂️ **80/20 stratified split** (`random_state=42`).

---

## 🏁 Results

| Model | Test accuracy |
|-------|--------------:|
| 🥇 Decision Tree | 99.89% |
| 🥈 Random Forest | 99.04% |
| 🧠 ANN (Keras, Dense + Dropout) | 96.8% |
| KNN | 96.71% |
| Logistic Regression | 95.63% |
| SVM | 95.57% |

The Decision Tree also reached **99.86%** mean accuracy in 5-fold cross-validation.

---

## ⚠️ Known limitation (important)

The very high accuracy is **not** a sign of a strong model. In this dataset every `main_genre` value belongs to exactly one type: genres such as *International TV Shows*, *Crime TV Shows* or *Kids' TV* only occur for TV shows, and *Dramas*, *Comedies* or *Documentaries* only for movies. Since `main_genre` is one of the selected features, it nearly **gives away the label** (target leakage), so the tree mostly reads the answer from that column.

**Planned fix:** remove `main_genre` (and re-check `num_genres`), retrain, and report the more honest accuracy. Features that do not encode the answer, such as `release_year`, `rating`, `main_country` and the cast and director counts, are better candidates.

---

## ▶️ How to run

```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn jupyter
jupyter notebook
```

1. Open `Netflix Content Type Classifier.ipynb` and run all cells (the paths may need adjusting to your folder).
2. Open `02_load_model_and_predict.ipynb` to reload the saved model and predict on sample rows.

---

## 🔮 Ideas for next steps

- Fix the leakage above and compare honest scores.
- Add text features from `description` (TF-IDF).
- Build a small web demo (for example with Streamlit) that loads the saved model.

---

## 👤 Author

**Issac Saeed Issac Ayad** · AI & Computer Science student, Minia National University
[GitHub](https://github.com/IssacSaeed-git) · [LinkedIn](https://www.linkedin.com/in/issac-saeed-63b0613b8)
