
# Question Answering with BERT (SQuAD)

This project implements a **question answering system** using a **BERT model** fine-tuned on the **SQuAD dataset**. It extracts answers from passages for given questions, demonstrating how transformer-based architectures can handle machine reading comprehension tasks.


## Overview

* **Goal**: Automatically answer questions from provided passages
* **Dataset**: SQuAD (Stanford Question Answering Dataset)
* **Model**: BERT-base (uncased) with a QA head
* **Approach**:

  * Tokenize questions and contexts
  * Map answers to token positions
  * Fine-tune BERT using Hugging Face `Trainer`
  * Evaluate with Exact Match (EM) and F1 metrics


## Demo Mode

For speed in Colab (internship presentation/demo):

* **Training**: 2,000 samples, 1 epoch
* **Evaluation**: 500 samples
* **Result**: Basic predictions (low accuracy)

### How to Improve Accuracy

* Train on **entire dataset** (87k samples)
* Increase to **2–3 epochs**
* Use **max sequence length = 512**
* Run on GPU (Colab Pro / Kaggle / local GPU)


## Results (Demo)

* **Exact Match (EM)**: \~20–25% (demo subset)
* **F1 Score**: \~35–40% (demo subset)
* Full training achieves \~80+% F1 (benchmark)


## How to Run

1. Open `qa_system.ipynb` in Google Colab
2. Run all cells to:

   * Load SQuAD dataset
   * Preprocess data
   * Fine-tune BERT model
   * Evaluate performance
   * Answer custom questions



## Example Prediction

```python
context = """
BERT is a transformer-based model developed by Google.
It achieved state-of-the-art results on multiple NLP tasks such as question answering and sentiment analysis.
"""

print(qa_pipeline(question="Who developed BERT?", context=context))
# Expected: 'Google'

print(qa_pipeline(question="What tasks did BERT perform well on?", context=context))
# Expected: 'question answering and sentiment analysis'
```

---



## Future Work

* Train with **entire dataset** for higher accuracy
* Add **web demo (Gradio/Streamlit)** for interactive QA
* Explore **DistilBERT** or **ALBERT** for lightweight deployment
* Experiment with **domain-specific QA datasets** (e.g., medical, legal)

---

### **Requirements**

```
transformers
datasets
evaluate
```
