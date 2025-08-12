# 🏨 Hotel Booking Cancellation Predictor

This repository contains a **machine learning project** to predict hotel booking cancellations using historical booking data.  
It includes:
- A trained model (`cancellation_model.pkl`)
- A Streamlit web app (`streamlit.py`) for easy prediction
- The dataset (`data_hotel_booking_demand.csv`)
- All required dependencies (`requirements.txt`)

## 📌 Project Overview
Hotel booking cancellations can cause significant revenue loss.  
By predicting potential cancellations before they happen, hotels can take **proactive measures** such as requiring deposits, offering incentives, or contacting the guest directly.

This project:
1. Preprocesses and cleans hotel booking data
2. Trains a machine learning classification model
3. Deploys a Streamlit app for real-time predictions

## 📊 Dataset
**File:** `data_hotel_booking_demand.csv`  
The dataset contains historical hotel booking records with:
- Guest demographics (e.g., `country`)
- Booking details (`market_segment`, `deposit_type`, etc.)
- Stay details (`total_of_special_requests`, `days_in_waiting_list`, etc.)
- Target variable: `is_canceled` (0 = not canceled, 1 = canceled)

## 🛠 Model Training
- **Algorithm tested:** Logistic Regression, KNN, Random Forest, XGBoost
- **Best model:** **K-Nearest Neighbors (KNN)** — chosen for highest F2-score
- **Evaluation metric:** F2-score (prioritizes Recall to catch more cancellations)
- **Recall:** 82% — detects most cancellations  
- **Precision:** 41% — some false positives, acceptable for low-cost interventions

## 🌐 Streamlit App
The `streamlit.py` script provides a simple interface:
1. Input booking details via the sidebar
2. Click **Predict Cancellation**
3. View probability of cancellation and risk classification

### Example App Features:
- Dropdowns for categorical fields (`market_segment`, `customer_type`, etc.)
- Sliders for numerical inputs (`previous_cancellations`, `booking_changes`, etc.)
- Probability-based decision threshold

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/hotel-booking-cancellation-prediction.git
cd hotel-booking-cancellation-prediction
