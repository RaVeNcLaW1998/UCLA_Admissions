# 🎓 UCLA Admissions Prediction

A machine learning web application that predicts the likelihood of admission into UCLA Master's programs using academic metrics. Built with **Streamlit**, modularized with reusable components, and trained using a **Neural Network (MLPClassifier)**.

---

## 🚀 Demo

> 🔗 https://uclaadmissions-ouxwfssqpazq8n3nw5eizd.streamlit.app

---

## 📊 Features

- Predict admission chances based on:
  - GRE & TOEFL scores
  - SOP & LOR strengths
  - CGPA
  - University rating
  - Research experience
- Built-in model training on first run (no separate training needed)
- Displays admission probability with:
  - ✅ Class prediction
  - 📈 Loss Curve during training
- Logs automatically stored to `logs/app.log`

---

## 🧠 Model

- **Model**: Multilayer Perceptron Classifier (MLPClassifier)
- **Accuracy**: ~90%
- **Auto-trained**: if no model exists (`admission_model.pkl` is generated automatically)

---

## 🗂 Project Structure

```
├── app.py                      # Streamlit app
├── ucla_admissions/
│   ├── config.py              # Paths, logging, and constants
│   ├── dataset.py            # Loads admission data
│   ├── features.py           # Preprocessing (training & user input)
│   ├── modeling/
│   │   ├── train.py          # Training, evaluation, load_or_train_model()
│   │   └── predict.py        # Predict from trained model
│   └── plots.py              # Optional: loss curve plotting
├── data/
│   └── raw/
│       └── Admission.csv     # Raw data
├── models/
│   └── admission_model.pkl   # Auto-generated trained model
├── logs/
│   └── app.log               # Logging output
├── requirements.txt
└── README.md
```

---

## 🛠 Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ucla-admissions-predictor.git
cd ucla-admissions-predictor
```

### 2. Create a virtual environment

```bash
python -m venv env
source env/bin/activate       # On Windows: .\env\Scriptsctivate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📈 Model Performance

You can check the model accuracy and loss curve by enabling the checkboxes in the UI.

Sample confusion matrix:

```
[[63  6]
 [ 4 27]]
Accuracy: 90%
```

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) — includes:

- `streamlit`
- `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`
- `joblib`

---

## 🧾 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Developed by [Athul Krishna Radhakrishnan Nair](https://github.com/RaVeNcLaW1998)