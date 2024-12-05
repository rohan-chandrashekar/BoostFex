# BoostFex

# 🌟 Detection of Bank Transaction Anomalies Using Gradient Boosted Federated Learning 🌟

![License](https://img.shields.io/badge/license-MIT-brightgreen) ![Issues](https://img.shields.io/github/issues/rohan-chandrashekar/BoostFex)

---

## 📚 Abstract

This repository hosts the official implementation of the research article:  
**"Detection of Bank Transaction Anomalies using Gradient Boosted Federated Learning"**  
Submitted to *IEEE Access*.  
Authors: **Rohan Chandrashekar, Rithvik Grandhi, Rahul Roshan G, Shylaja SS**  

### 🛠️ Overview
Fraudulent transactions pose critical challenges for the banking sector, affecting customer trust and financial security. In this research, we propose an innovative **Federated Learning framework** integrated with **Gradient Boosting algorithms** to enhance anomaly detection. Our approach ensures robust privacy-preserving mechanisms while delivering high accuracy on imbalanced datasets.

---

## 🔍 Key Features

- **Federated Learning Architecture**: Leverages decentralized training to enhance data security.
- **Gradient Boosted Models**: Optimized for imbalanced datasets and high-dimensional transaction data.
- **Privacy-Preserving Techniques**: Compliance with GDPR and CCPA standards.
- **Scalable and Modular Design**: Supports both cyclic and bagging federated learning models.

---

## 🏗️ Repository Structure

```
📁 src/                  # Implementation of different models
📁 Dataset/              # Dataset Generator and Preprocessing files
📁 Results/              # Raw result data and Graph Visualization
📁 Literature Survey     # Literature Survey Papers
📁 Journal Article       # Journal Article Submitted to IEEE Access
📄 README.md             # Project description
```

---

## 📈 Results Summary

| Metric           | Cyclic Model | Bagging Model | Centralized Baseline |
|-------------------|--------------|---------------|-----------------------|
| Accuracy          | 98%          | 97%           | 92%                   |
| Precision         | 97%          | 95%           | 93%                   |
| Recall            | 97%          | 96%           | 89%                   |
| F1-Score          | 97%          | 96%           | 91%                   |

For more details, refer to the *Results & Analysis* section of the article.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Recommended IDE: Visual Studio Code / PyCharm
- Required packages: Install dependencies using:


### Running the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/rohan-chandrashekar/BoostFex.git
   cd repositoryname
   ```

2. Prepare the dataset:
   Place your dataset in the `data/` directory and follow preprocessing steps outlined in `scripts/data_preparation.py`.

3. Train the models:
   - For Cyclic Model:
     ```bash
     python scripts/train_cyclic.py
     ```
   - For Bagging Model:
     ```bash
     python scripts/train_bagging.py
     ```

4. Evaluate performance:
   ```bash
   python scripts/evaluate.py
   ```

---

## 🤝 Contributing

We welcome contributions to improve the implementation or extend its capabilities. Please submit a pull request or open an issue for discussion.

---

## 📝 License

This repository is licensed under the MIT License. See `LICENSE` for more details.

---

## 📬 Contact

For queries, suggestions, or feedback:
- 📧 **[chandrashekar.rohans@gmail.com](mailto:chandrashekar.rohans@gmail.com)**  
- GitHub: [@yourusername](https://github.com/yourusername)

---
