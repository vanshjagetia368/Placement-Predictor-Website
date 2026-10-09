# Placement Predictor 🎓

A small Streamlit website built on the Logistic Regression model in `Basic_toy_project.ipynb`.
Enter an **IQ** and a **CGPA** and it predicts whether the student gets placed, with a probability,
the CGPA "tipping point", and a chart of how the odds change.

## Project files

| File | Purpose |
| --- | --- |
| `Basic_toy_project.ipynb` | Loads the data, scales it, trains the model, exports both `.pkl` files |
| `model.pkl` | Trained `LogisticRegression` |
| `scaler.pkl` | Fitted `StandardScaler` (inputs must be scaled exactly as in training) |
| `placement (1).csv` | Training data (100 students). Optional for the app: it only draws the chart on the "How it works" tab |
| `app.py` | The Streamlit website |
| `requirements.txt` | Python dependencies |

The last cell of the notebook saves both `model.pkl` and `scaler.pkl`. The feature order used in
training is **CGPA, IQ**; the website shows IQ first but sends them to the model in the right order.

## Important: keep the scikit-learn version pinned

`model.pkl` and `scaler.pkl` were created with **scikit-learn 1.6.1**, so `requirements.txt` pins that exact
version. Loading a pickle with a different version can warn or give wrong results. If you retrain in a newer
environment, update the pin to match `pip show scikit-learn`.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Create a GitHub repository, for example `placement-predictor`.
2. Upload every file listed above.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/), sign in with GitHub, and choose **Create app**.
4. Select the repository, branch, and `app.py`, then deploy.

GitHub Pages only hosts static files, so it cannot run this Python model. Streamlit Community Cloud can.

## Model caveat

The model learned from only 100 students, and the notebook's 90% accuracy comes from a 10-student test set.
In this dataset the model relies almost entirely on CGPA. This is a learning project, not a real-world
placement guarantee.
