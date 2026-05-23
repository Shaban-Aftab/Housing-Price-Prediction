# 🏡 Multimodal Housing Price Prediction (Visual + Tabular Late Fusion in PyTorch)

A modular, production-ready machine learning system that uses a **late-fusion neural network** implemented in native **PyTorch** to predict house values from both visual (frontal home listing images) and structured tabular metadata (bedrooms, bathrooms, area, zipcode). It features custom dataset structures, a pre-trained **MobileNetV2 CNN** visual branch, a multi-layer tabular MLP, and a Streamlit serving interface with real-time photo uploads.

🚀 **GitHub Repository:** [Shaban-Aftab/Scikit-learn-Pipeline](https://github.com/Shaban-Aftab/Scikit-learn-Pipeline) (Or your active multimodal repo)

---

## 🎨 Features & Architecture

* **Late-Fusion Multimodal Architecture**: Features decoupled visual and structured tabular extraction pathways in PyTorch (`nn.Module`), concatenated into a single `128`-dimensional fused representation before estimation.
* **Lightweight Visual CNN (MobileNetV2)**: Customizes a pre-trained **MobileNetV2** backbone, freezing initial layers to prevent overfitting on small datasets while training a custom classification head to map visual features to `64` embedding dimensions.
* **Tabular Feature MLP**: Standard-scales square footage and room counts, applies one-hot zipcode encoding, and maps features through an MLP with batch normalization, ReLU activation, and dropout to `64` embedding dimensions.
* **Streamlit Serving Dashboard**: Supports drag-and-drop property image uploading, interactive feature forms, and self-healing model checkpointers.

---

## 📂 Project Structure

```
├── multimodal_housing_price.ipynb  # Self-contained Jupyter Notebook for Kaggle execution
├── pipeline.py                     # Custom HousingDataset loader & StandardScaler exports
├── model.py                        # Late-Fusion visual-tabular network definition
├── train.py                        # GPU/CPU backprop training loop & metrics logging
├── app.py                          # Streamlit interactive uploader UI dashboard
├── requirements.txt                # Dependencies file
└── README.md                       # Documentation
```

---

## ⚡ Setup & Serving

### 1. Local Installation
Clone the repository and install all required scientific, deep learning, and web dependencies:

```bash
git clone <your-repo-url>
cd "Housing Price Prediction"
pip install -r requirements.txt
```

### 2. Streamlit Web Dashboard
Launch the servable dashboard locally or deploy it to Streamlit Cloud:

```bash
streamlit run app.py
```

* **Note**: Download the trained model weights (`multimodal_house_model.pth`) and scaler checkpoint (`tabular_scaler.joblib`) from your Kaggle notebook run, and drop them into the project root directory to serve predictions instantly!

---

## 🎯 Kaggle Integration

Because neural network training requires GPU acceleration, you should run the **`multimodal_housing_price.ipynb`** notebook directly on Kaggle:

1. **Create a Kaggle Notebook** and import `multimodal_housing_price.ipynb`.
2. **Execute all cells sequentially**. The notebook will automatically clone the dataset from GitHub (`git clone https://github.com/emanhamed/Houses-dataset`), preprocess it, train the PyTorch network, print performance stats, and export the models.
3. **Download** the generated files:
   * `multimodal_house_model.pth`
   * `tabular_scaler.joblib`
4. **Push** them to your GitHub repository to enable instant serving on Streamlit Cloud!
