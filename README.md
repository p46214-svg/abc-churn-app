# ABC Ltd. – Customer Churn & Monthly Charge Predictor

A Streamlit app that uses the two models from `ABC_Ltd_Predictive_Analytics.ipynb`:

| File | Model | Predicts |
|---|---|---|
| `models/logistic_churn_model.pkl` | Logistic Regression | Churn probability + risk band |
| `models/linear_monthly_charge_model.pkl` | Linear Regression | Expected monthly charge |

## Folder structure (upload exactly like this)

```
abc-churn-app/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── logistic_churn_model.pkl
    └── linear_monthly_charge_model.pkl
```

## Deploy on Streamlit Community Cloud

### Step 1 – Put the files on GitHub
1. Sign in at https://github.com and click **New repository**.
2. Name it, for example `abc-churn-app`, set it to **Public**, and click **Create repository**.
3. On the empty repo page click **uploading an existing file**.
4. Drag in `app.py`, `requirements.txt`, `README.md` **and the whole `models` folder**
   (drag the folder itself so the files end up inside `models/`).
5. Click **Commit changes**.
6. Check that the repo shows a `models` folder with both `.pkl` files inside it.

### Step 2 – Deploy
1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `your-username/abc-churn-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** choose a name (optional)
4. Open **Advanced settings** and set **Python version = 3.12**. Click **Save**.
5. Click **Deploy**. The first build takes 2–5 minutes.

### Step 3 – Share
Copy the `https://<your-name>.streamlit.app` link from the browser.

## Run on your own computer (optional)
```
pip install -r requirements.txt
streamlit run app.py
```

## Why the versions are pinned
The models were saved in Colab with **scikit-learn 1.6.1** and NumPy 2.x.
A `.pkl` file only reliably loads on the same scikit-learn version, so
`requirements.txt` pins `scikit-learn==1.6.1`. Do not remove the pin.
`flask` was removed from the original requirements because the app does not use it.

## Common errors

| Error on Streamlit Cloud | Fix |
|---|---|
| `Model files not found` | The `models` folder is missing or the files sit in the repo root. Move both `.pkl` files into `models/`. |
| `ModuleNotFoundError: numpy._core` | NumPy 1.x was installed. Keep `numpy==2.2.6` in requirements. |
| `InconsistentVersionWarning` / attribute errors when loading | scikit-learn version changed. Keep `scikit-learn==1.6.1`. |
| Build fails compiling scikit-learn / pandas | Python version too new. In app **Settings → General**, pick Python 3.12 (you may need to delete and redeploy the app to change it). |
| Changes not showing | Commit to GitHub; the app redeploys automatically. Use **⋮ → Reboot app** if needed. |
