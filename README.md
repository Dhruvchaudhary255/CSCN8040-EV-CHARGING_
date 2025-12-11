# CSCN8040-EV-CHARGING_

---

# ⚡ **Rural EV Charging Gap Analysis & RDI Modeling**

### *CSCN8040 – Machine Learning for Data Analytics*

### **Conestoga College**

---

## 👥 **Team Members**

| Role            | Name                        |
| --------------- | --------------------------- |
| **Team Leader** | Dhruv Bhanuprasad Chaudhary |
| **Team Member** | Manpreet Kaur               |
| **Team Member** | Abdul Bari Mohammad         |
| **Team Member** | Vishal Mukeshbhai Shah      |

---

# 📌 **Project Overview**

Canada faces a **significant EV charging gap**, especially in *rural areas* where infrastructure investment and forecasting models fall short.
This project solves that gap by:

### ✅ Building a **Rural Demand Index (RDI)** using:

* EV counts
* Charging station availability
* City demographics
* Population density
* Accessibility indicators

### ✅ Developing a **predictive regression model**

Performs with extremely high accuracy:

```
MAE: 0.000108  
RMSE: 0.000112  
R² Score: 0.9999987
```

### ✅ Creating an ML pipeline integrated with:

* Data wrangling
* EDA
* Feature engineering
* Statistical testing
* Model training
* Streamlit visualization dashboard

This repository contains all required datasets, scripts, notebooks, and the project app.

---

# 📁 **Repository Structure**

```
CSCN8040-EV-CHARGING_/
│
├── DataSets/
│   ├── ev_city_station_summary.csv
│   ├── canadacities.csv
│
├── project.ipynb             # Full EDA + RDI + Model notebook
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Required libraries
├── README.md                 # (This file)
```

---

# ⚙️ **Installation & Setup**

### **1. Clone the repository**

```bash
git clone https://github.com/Dhruvchaudhary255/CSCN8040-EV-CHARGING_.git
cd CSCN8040-EV-CHARGING_
```

### **2. Create a virtual environment (recommended)**

```bash
python -m venv env
source env/bin/activate   # Mac/Linux
env\Scripts\activate      # Windows
```

### **3. Install required dependencies**

```bash
pip install -r requirements.txt
```

---

# ▶️ **How to Run the Streamlit App**

```bash
streamlit run app.py
```

This opens the dashboard in your browser automatically.

---

#  **Key Features**

### 🔍 **1. Exploratory Data Analysis (EDA)**

* Missing values handled
* Distribution plots
* Correlation analysis
* Population density & EV adoption insights

###  **2. Rural Demand Index (RDI) Calculation**

* Normalized multi-factor scoring
* Weighted index measuring infrastructure need
* Identification of top underserved regions

###  **3. Machine Learning Model**

* Linear Regression (best-performing model)
* Predicts charging demand using engineered features
* Achieves near-perfect accuracy (R² > 0.9999)

### 🧪 **4. Statistical Testing**

* ANOVA test to validate data differences across provinces

### 💻 **5. Streamlit Dashboard**

Displays:

* City demographics
* RDI scores
* Model output
* Top underserved cities

---

# 📈 **Model Performance**

```
Model Performance:
MAE: 0.0001080033
RMSE: 0.0001125345
R2 Score: 0.999998769
```

✔️ Extremely low error
✔️ Excellent predictive stability

---

# 🌐 **RDI Framework**

RDI = (EV_per_capita + Distance Score + Accessibility Score) / 3

This helps governments & utility providers prioritize **rural areas needing urgent infrastructure deployment**.

---

# 📸 **Screenshots**

### **Model Results**

![Model Performance](https://raw.githubusercontent.com/Dhruvchaudhary255/CSCN8040-EV-CHARGING_/main/assets/model_performance.png)

*(Add your local images to /assets folder for GitHub display — I can generate final PNG files for you if needed.)*

---

# 🧠 **What This Project Demonstrates**

✔ Ability to perform full-scope data analytics
✔ Capability to build ML pipelines
✔ Deployment-ready Streamlit application
✔ Strong understanding of forecasting and optimization
✔ Framework aligned with TBP, OMT, and MLOps principles

---

# 🔮 **Future Improvements**

* Integrate real-time traffic & mobility data
* Include weather & economic indicators
* Use XGBoost or Random Forest for deeper prediction accuracy
* Add geospatial visualization (Folium / Kepler.gl)
* Provincial-level optimization heatmap

---

# 📜 **License**

This project is for academic use under **Conestoga College CSCN8040**.
Unauthorized commercial use is not permitted.

---

# 🙌 **Acknowledgments**

Special thanks to
**Professor Maria Wesolowski**
for guidance on TBP, A3 Forms, OMT scoring, and ML methodology.

---


