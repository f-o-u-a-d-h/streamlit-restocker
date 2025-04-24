# 🏥 Hospital Medicine Restocking Predictor

## 📘 Introduction
This project is a machine learning-powered web application designed to help hospitals optimize their medicine restocking process. It allows hospital staff to input current usage statistics and inventory levels and receive a recommendation on how much medicine to restock. This is particularly useful in minimizing waste, preventing shortages, and managing hospital logistics efficiently.

The underlying model was trained on **synthetic hospital inventory data** that was generated to simulate realistic demand patterns for a high-demand generic drug. The simulated dataset was crafted to reflect variations in hospital department visits, consumption trends, and inventory dynamics across time.

Given the tabular nature of the data and the presence of both numerical and categorical features, **LightGBM** was selected for modeling due to its:
- Strong performance on structured datasets
- Built-in handling of categorical variables
- Speed and efficiency, especially with large feature sets and high-dimensional data

The model was trained offline and saved as `restocking_model.pkl`. It is loaded during inference and does not require retraining in the deployed application.

The app is deployed using Streamlit Cloud for easy access, interaction, and demonstration.

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- pip

### Local Setup
1. Clone the repository:
```bash
git clone https://github.com/f-o-u-a-d-h/streamlit-restocker.git
cd streamlit-restocker
```
2. Install the dependencies:
```bash
pip install streamlit pandas joblib lightgbm numpy scikit-learn
```
3. Run the app:
```bash
streamlit run restock_app.py
```
4. Open your browser and visit:
```
http://localhost:8501
```

---

## 🚀 Usage

### Web App Interface
After launching, you will see a form to enter:
- Date
- Number of outpatient visits (max 2,500)
- Number of emergency visits (max 250)
- Number of inpatient admissions (max 250)
- Quantity of medicine consumed (max 3,000 tablets)
- Quantity of medicine currently in stock (max 5,000 tablets)

Click the **Predict** button and the app will display the ideal quantity of medicine to restock.

### Docker Usage (optional)
You can also run this app in a Docker container.

1. Build the Docker image:
```bash
docker build -t streamlit-restocker .
```
2. Run the container:
```bash
docker run -p 8501:8501 streamlit-restocker
```
3. Access the app at:
```
http://localhost:8501
```

### Streamlit Cloud Deployment
To deploy this app using Streamlit Cloud:
1. Push this project to a public GitHub repository
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **"New App"** and connect your GitHub repo
4. Choose:
   - Repository: `f-o-u-a-d-h/streamlit-restocker`
   - Branch: `main`
   - App file: `restock_app.py`
5. Click **"Deploy"** and your app will be live.
Deployed Example: https://app-restocker-b672wwylekojh23gghvurq.streamlit.app/

---

## 📂 Project Structure
```
restock_app.py              # Main Streamlit app
predictor.py                # ML prediction logic
restocking_model.pkl        # Trained LightGBM model (pretrained and saved)
Dockerfile                  # For Docker-based deployment
.dockerignore               # Docker ignore rules
.gitignore                  # Git ignore rules
README.md                   # Project documentation
requirements.txt            # Optional dependency list
```

---

## 📫 Contact
For questions or contributions, please reach out via GitHub Issues.
