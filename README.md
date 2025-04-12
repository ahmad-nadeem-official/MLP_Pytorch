📱 Mobile Price Prediction using ANN (PyTorch)
==============================================

Welcome to one of my cleanest **PyTorch projects**!  
This repo uses an **Artificial Neural Network (ANN)**, specifically a **Multilayer Perceptron (MLP)**, to predict mobile phone price ranges based on specs like RAM, battery, processor, etc. in this code I didn't clean the data as well as visualized because this is the same dataset which is used in prevoius project(I have done cleaning and visualization in prevoius practice) here([Open in Google Colab](https://colab.research.google.com/drive/1Enk9PC1ikD2g6EZvQdPJ5RsrVHFmtVIr#scrollTo=qjG0yHtjRG3q)).

> 🎯 Achieved **~93% Accuracy** with a simple ANN architecture!

* * *

🔗 Related Project
------------------

Check out my earlier 🔧 **Neural Network From Scratch** where I didn't even use `nn.Module`, `CrossEntropyLoss`, or any PyTorch shortcut.  
👉 🧠 Neural Network from Scratch, as well as loss function you can see the code of handmade neural network from here([Open in Google Colab](https://colab.research.google.com/drive/1Enk9PC1ikD2g6EZvQdPJ5RsrVHFmtVIr#scrollTo=qjG0yHtjRG3q))

* * *

📂 Project Overview
-------------------

This repo includes:

*   A clean PyTorch-based pipeline for tabular classification
    
*   ANN using `nn.Sequential` with `BatchNorm`, `Dropout`, `ReLU`
    
*   Model training + evaluation
    
*   Custom `Dataset` class
    
*   Accuracy reporting
    

* * *

🧠 What is ANN / MLP?
---------------------

An **Artificial Neural Network (ANN)** is inspired by the human brain — it consists of **neurons (layers)** that process input data and pass it forward.  
A **Multilayer Perceptron (MLP)** is a type of ANN with:

*   An **input layer** (features)
    
*   One or more **hidden layers** (ReLU, Dropout, BatchNorm, etc.)
    
*   An **output layer** (for classification)
    

It’s perfect for predicting mobile price range from structured/tabular data.

* * *

🚀 Getting Started
------------------

> ✨ Recommended: Use **Google Colab** to run this notebook effortlessly.

### 📌 Colab Notebook

👉 🚀 Run on Google Colab [Open in Google Colab](https://colab.research.google.com/drive/1SbXUIGewAaA85huMvGBhJebBNkmdw_SH#scrollTo=T9qDAX2SRjsn)

* * *

### 💻 Technical Stack
------------------

![Python](https://img.shields.io/badge/Python-3.9%252B-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)  
![NumPy](https://img.shields.io/badge/NumPy-1.24-yellow)  
![Pandas](https://img.shields.io/badge/Pandas-1.5-orange)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-green)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-blueviolet)  
![Google-collab](https://img.shields.io/badge/google-colab-orange)
    

* * *

🧾 Installation & Usage Guidelines in VS-Code
----------------------------------

### 🧱 Clone the Repo


`git clone https://github.com/ahmad-nadeem-official/MLP_Pytorch.git`
-----------------------------------
`cd MLP_Pytorch/mlp` 

### 🧪 Requirements

If you're running locally, install the following:


`pip install torch
pip install pandas numpy scikit-learn matplotlib seaborn` 

### 🧠 VS Code Users

If you're using **VS Code**:

1.  Open folder in VS Code
    
2.  Make sure you select your Python environment (`Ctrl + Shift + P` → Python: Select Interpreter)
    
3.  Recommended to use a virtual environment
    
4.  Paste your `train.csv` and `test.csv` in the project folder
    
5.  Run the script: `python train.py` or run each cell if using `.ipynb`
    

* * *

⚙️ Project Flow
---------------

python

CopyEdit

`📦 Load CSV Data
🔍 Preprocess and Split
📊 Custom Dataset + Dataloader
🧠 Build Model with nn.Sequential
🎯 Train with CrossEntropy + Adam
📈 Evaluate Model Accuracy` 

* * *

📁 Dataset Info
---------------

The dataset used for training has:

*   🔢 Shape: `(2000, 21)`
    
*   📌 Target: `price_range` (0 to 3)
    
*   📉 Features: RAM, Battery, Processor, Storage, etc.
    

* * *

💡 Pro Tips
-----------

*   Tweak **hidden layer sizes**, **batch size**, **learning rate**, and **dropout** for better performance.
    
*   Try **LeakyReLU**, **LayerNorm**, or **different optimizers**.
    
*   You can save this model using `torch.save()` and load later for production or Flask/Streamlit apps.
    

* * *

✅ Final Accuracy
----------------

Achieved **~93%** test accuracy with just 50 epochs using default settings on GPU.  
That’s 🔥 for a simple ANN on structured data!