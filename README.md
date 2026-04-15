# Customer Churn Prediction

## 🎯 Project Overview

This project aims to predict customer churn using machine learning techniques. The main goal is to **identify customers at risk of leaving (churn)**, prioritizing **high recall** to minimize missed churners.

The project follows a complete data science workflow, from data cleaning and exploratory analysis to model building, evaluation, and business interpretation.

---

## 📊 Dataset

* **Dataset:** Telco Customer Churn
* **Size:** ~7,000 customers
* **Features:** 21 variables including:

  * Customer demographics
  * Contract type
  * Internet services
  * Payment methods
  * Charges and tenure

---

## 🔍 Exploratory Data Analysis (EDA)

Key patterns observed:

* Customers with **month-to-month contracts** show the highest churn rates
* **Long-term contracts (1–2 years)** significantly reduce churn
* Customers using **fiber optic internet** tend to churn more
* **New customers (low tenure)** are more likely to churn
* Customers with **lower total charges** are more likely to leave

---

## ⚙️ Data Preprocessing

* Converted `TotalCharges` to numeric (handling invalid values)
* Dropped missing values
* Applied **one-hot encoding** for categorical variables
* Split data into training, validation, and testing sets (stratified)

---

## 🤖 Models Used

### 1. Logistic Regression

* Baseline model
* Applied **feature scaling**
* Handled class imbalance using:

  * `class_weight="balanced"`
  * threshold tuning

---

### 2. XGBoost

* Tree-based model (no scaling required)
* Handled class imbalance using:

  * `scale_pos_weight`
* Captures non-linear relationships

---

## 📈 Model Performance

| Model                       | Recall (Churn) | Precision | Accuracy |
| --------------------------- | -------------- | --------- | -------- |
| Logistic Regression (tuned) | ~0.76          | ~0.52     | ~0.75    |
| XGBoost (balanced)          | ~0.76          | ~0.49     | ~0.73    |

---

## 🎯 Key Decision: Threshold Tuning

Instead of using the default classification threshold (0.5), the threshold was selected on a validation set and applied once to the test set to:

* **Increase recall**
* Detect more churners
* Accept a higher number of false positives

This reflects a real-world business trade-off:

> Missing a churner is more costly than incorrectly flagging a loyal customer.

---

## 📊 Model Evaluation

* ROC Curve and AUC used to evaluate model performance
* Threshold optimization performed using Youden’s Index on the validation set

---

## 🔝 Feature Importance (XGBoost)

Top drivers of churn:

* Contract type (most important factor)
* Internet service (fiber optic)
* Payment method (electronic check)
* Tenure (customer lifetime)
* Streaming services usage

---

## 🧠 Key Insights

* **Contract length is the strongest predictor of churn**
* Customers on **month-to-month plans** are the highest risk group
* **Long-term contracts improve retention significantly**
* **Fiber optic users show higher churn behavior**
* **Customer tenure is critical**: newer customers churn more
* Coefficients from Logistic Regression are interpreted as direction and relative strength because features are standardized

---

## 💼 Business Impact

This model can help companies:

* Identify high-risk customers early
* Design targeted retention strategies
* Offer incentives for long-term contracts
* Reduce customer acquisition costs by improving retention

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn

---

## 🚀 Future Improvements

* Hyperparameter tuning (Grid Search / Random Search)
* Try advanced models (LightGBM, CatBoost)
* Deploy model as an API
* Build a dashboard for business users

---

## 📌 Conclusion

This project demonstrates how machine learning can be applied to a real-world business problem, combining:

* Data analysis
* Predictive modeling
* Model evaluation
* Business-oriented insights

---
