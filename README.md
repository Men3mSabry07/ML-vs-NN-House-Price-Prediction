# 🏠 House Price Prediction — Machine Learning vs Neural Networks

## 📌 Overview
This project focuses on predicting house prices using both **Machine Learning** and **Neural Networks** using the **same dataset** and the **same preprocessing pipeline**.

The main goal of this project was to explore the practical differences between traditional Machine Learning models and Deep Learning models for a real-world regression problem.

The project uses approximately **100,000 housing records** and compares model performance using common regression metrics such as:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# Project Idea

Instead of directly choosing one approach, I wanted to answer the following question:

> “Which performs better for structured/tabular house pricing data: traditional Machine Learning or Neural Networks?”

To ensure a fair comparison:

✅ Same dataset  
✅ Same preprocessing  
✅ Same train/test split  
✅ Same evaluation metrics  

---

# 📂 Repository Structure

```bash
├── Data
│   ├── Cleaned.csv
│   └── Orig_Data.csv
│
├── comparison
│   └── ML_vs_NN_Comparison.ipynb
│
├── machine_learning
│   └── Regression_Models.ipynb
│
├── neural_network
│   ├── neural_network_PyTorch.ipynb
│   └── neural_network_tensorflow.ipynb
│
├── preprocessing
│   └── Preprocessing.ipynb
│
├── results
│   ├── graphs
│   │   ├── experiments_table.png
│   │   ├── r2_MSE_comparison.png
│   │   └── train_val_loss.png
│   │
│   ├── results_ml.csv
│   └── results_nn.csv
│
└── README.md
```

---

# 📊 Dataset Information

- Dataset Size: ~100,000 records
- Problem Type: Regression
- Target Variable: `price_usd_log`

### Main Features
- House type
- Region
- Area
- Number of rooms
- House size (sqm)
- House age
- Sales type
- Mortgage bond yield
- Quarter / Year information

---

# ⚙️ Data Preprocessing

The preprocessing pipeline included:

## 1️⃣ Handling Missing Values
- Removed invalid entries
- Dropped rows containing missing values

---

## 2️⃣ Cleaning Numerical Features

Invalid values were replaced with `NaN`:
- `year_build <= 0`
- `sqm <= 0`
- `no_rooms <= 0`

---

## 3️⃣ Feature Engineering

### House Age Feature

```python
house_age = CURRENT_YEAR - year_build
```

### Quarter Processing

The `quarter` feature was split into:
- `year`
- `quarter_num`

---

## 4️⃣ Log Transformation

House prices were highly skewed, so log transformation was applied:

```python
df["price_usd_log"] = np.log1p(df["price_usd"])
```

This helped normalize the distribution and improve model learning.

---

## 5️⃣ Encoding Categorical Variables

One-Hot Encoding was applied using:

```python
pd.get_dummies(..., drop_first=True)
```

---

## 6️⃣ Feature Scaling

Standardization was applied before training Neural Networks.

---

# 🤖 Machine Learning Models

The following regression models were implemented:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

---

# 🧠 Neural Network Models

Multiple MLP (Multilayer Perceptron) architectures were implemented using **PyTorch**.

---

## Model A — ReLU Network

### Architecture
- 256 → 128 → 64 → 32
- Activation Function: ReLU

### Techniques Used
- Huber Loss
- Adam Optimizer
- Batch Normalization
- Dropout
- Early Stopping
- Learning Rate Scheduler

---

## Model B — Tanh Network

Same architecture as Model A but using:

```python
nn.Tanh()
```

instead of ReLU activation.

---

## Model C — Deep ReLU Network

### Larger Architecture
- 512 → 256 → 128 → 64 → 32

### Lower Learning Rate
```python
lr = 0.0005
```

This model achieved the best overall performance.

---

# 📈 Training Techniques

The Neural Network training pipeline included:

- Mini-batch training
- Early stopping
- Learning rate scheduling
- Batch normalization
- Dropout regularization
- Huber Loss for robust regression

---

# 📉 Model Performance

## Final Comparison

| Model | MSE | RMSE | MAE | R² Score |
|---|---|---|---|---|
| Linear Regression | 0.3577 | 0.5981 | — | 0.4072 |
| Decision Tree | 0.3074 | 0.5545 | — | 0.4905 |
| Random Forest | 0.2680 | 0.5176 | — | 0.5559 |
| Neural Network (MLP) | 0.2620 | 0.5119 | 0.3816 | **0.5658** |

---

# 🏆 Best Model

## Deep ReLU Neural Network (Model C)

### Results
- MSE: **0.2605**
- MAE: **0.3805**
- R² Score: **0.5683**

The Neural Network slightly outperformed traditional Machine Learning models on this structured dataset.

---

# 📊 Visualizations

The project includes:
- Training vs Validation Loss
- MSE vs R² Comparison
- Experiments Comparison Table

Located inside:

```bash
results/graphs/
```

# 📌 Key Learnings

Through this project, I learned:

- Differences between Machine Learning and Deep Learning
- How preprocessing impacts model performance
- Importance of feature engineering
- How regularization improves Neural Networks
- Why Random Forest performs strongly on tabular data
- How to tune Neural Network architectures

---

# 📬 Future Improvements

Possible future improvements:
- Hyperparameter tuning with Optuna
- XGBoost / LightGBM comparison
- Cross-validation
- Model deployment using Flask or FastAPI
- Building an interactive web application
---

```markdown id="vw9x7y"
# 👨‍💻 Author

Developed by Abdelmenem Sabry

A practical comparison between Machine Learning and Neural Networks for house price prediction.
```


# ⭐ Final Conclusion

For structured house pricing data:

- Traditional Machine Learning models performed very well.
- Random Forest was highly competitive.
- Neural Networks achieved the best overall performance after careful tuning.

This project demonstrates that:

> Deep Learning is not always dramatically better for tabular data, but with proper architecture and regularization, it can outperform classical Machine Learning models.
