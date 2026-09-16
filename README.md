# ⚖️ BERT-Based Multi-Task Legal AI System

An end-to-end **Legal AI platform** designed to analyze legal documents and assist with multiple legal NLP tasks, including **case outcome prediction, legal entity extraction, document summarization, and risk assessment**.

The system uses **BERT-based Transformer models, PyTorch, and NLP pipelines** to process legal documents and generate task-specific predictions from uploaded PDF files.

---

## 🚀 Features

### 📄 Legal Document Processing

* Upload and process legal documents in PDF format.
* Extract relevant text from documents.
* Prepare and preprocess legal text for downstream NLP tasks.

### 🔮 Case Outcome Prediction

* Analyzes the content of a legal document.
* Uses a BERT-based NLP model to generate predictions related to case outcomes.
* Provides model-generated predictions through the application interface.

### 🏷️ Legal Entity Extraction

Identifies important entities and information from legal documents, such as:

* People
* Organizations
* Locations
* Courts
* Legal references
* Other relevant legal entities

### 📝 Document Summarization

* Generates concise summaries of lengthy legal documents.
* Helps reduce the time required to review large documents.
* Uses Transformer-based NLP techniques for text understanding.

### ⚠️ Legal Risk Assessment

* Analyzes extracted legal information.
* Identifies potentially relevant risk indicators.
* Provides an automated risk assessment based on model outputs.

---

## 🧠 AI & NLP Architecture

The system follows a multi-stage document analysis pipeline:

```text
                ┌─────────────────────┐
                │    Legal PDF        │
                │     Upload          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Document Ingestion  │
                │ & Text Extraction   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Text Preprocessing  │
                │ & Tokenization      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ BERT / Transformer  │
                │      Models         │
                └──────────┬──────────┘
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
          ▼                ▼                 ▼
    ┌───────────┐    ┌────────────┐    ┌────────────┐
    │  Outcome  │    │   Entity   │    │ Summarizer │
    │ Prediction│    │ Extraction │    │            │
    └───────────┘    └────────────┘    └────────────┘
          │                │                 │
          └────────────────┼─────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Risk Assessment│
                  └─────────────────┘
```

---

## 🛠️ Tech Stack

| Technology                    | Purpose                                         |
| ----------------------------- | ----------------------------------------------- |
| **Python**                    | Core development and ML pipelines               |
| **PyTorch**                   | Deep learning and model inference               |
| **BERT**                      | Transformer-based legal text understanding      |
| **Hugging Face Transformers** | Pre-trained Transformer models and tokenization |
| **NLP**                       | Text processing and legal document analysis     |
| **Flask**                     | Backend API and model-serving layer             |
| **PDF Processing**            | Legal document ingestion and text extraction    |

---

## 🧩 Multi-Task Learning

The platform is designed around multiple legal NLP tasks rather than a single prediction model.

### Task 1 — Case Outcome Prediction

```text
Legal Document
      ↓
Text Extraction
      ↓
BERT Tokenization
      ↓
Transformer Model
      ↓
Outcome Prediction
```

### Task 2 — Entity Extraction

```text
Legal Text
    ↓
BERT / NLP Model
    ↓
Token Classification
    ↓
Legal Entities
```

### Task 3 — Document Summarization

```text
Legal Document
      ↓
Text Processing
      ↓
Transformer Model
      ↓
Concise Summary
```

### Task 4 — Risk Assessment

```text
Legal Document
      ↓
Feature Extraction
      ↓
Model Inference
      ↓
Risk Indicators
      ↓
Risk Assessment
```

---

## 🏗️ System Architecture

The application follows a modular architecture:

```text
Frontend / Client
       │
       ▼
   Flask API
       │
       ▼
Document Processing
       │
       ├──────────────► PDF Text Extraction
       │
       ▼
 NLP Preprocessing
       │
       ▼
BERT / Transformer Layer
       │
       ├──► Outcome Prediction
       ├──► Entity Extraction
       ├──► Summarization
       └──► Risk Assessment
       │
       ▼
   Model Results
       │
       ▼
   API Response
```

This separation allows individual NLP components to be developed, tested, and improved independently.

---

## 📂 Project Structure

A typical structure for the project is:

```text
Multi-Task-Legal-AI/
│
├── app/
│   ├── routes/
│   ├── services/
│   └── utils/
│
├── models/
│   ├── outcome_prediction/
│   ├── entity_extraction/
│   ├── summarization/
│   └── risk_assessment/
│
├── data/
│
├── preprocessing/
│   ├── pdf_processor.py
│   └── text_processor.py
│
├── inference/
│   └── inference_pipeline.py
│
├── requirements.txt
├── app.py
└── README.md
```

> The exact directory structure may differ depending on the implementation.

---

## 🔄 Workflow

1. **Upload** a legal PDF document.
2. **Extract** text from the document.
3. **Preprocess** and tokenize the extracted text.
4. **Encode** the text using BERT/Transformer models.
5. **Run inference** for the required legal NLP tasks.
6. **Generate results** for:

   * Case outcome prediction
   * Entity extraction
   * Document summarization
   * Risk assessment
7. **Return results** through the Flask API.

---

## 🧪 Model Pipeline

The core NLP pipeline uses **BERT-based Transformer architecture**.

```text
Input Legal Text
       ↓
Tokenizer
       ↓
Input IDs + Attention Mask
       ↓
BERT Transformer
       ↓
Contextual Representations
       ↓
Task-Specific Prediction Heads
       ↓
Legal AI Outputs
```

BERT provides contextual representations of legal text, which can then be used by different task-specific components.

---

## 🔌 Flask API

Flask provides the backend layer responsible for:

* Receiving legal document uploads
* Processing requests
* Running model inference
* Returning prediction results
* Connecting the NLP pipeline with the application interface

Example API workflow:

```text
POST /analyze
       │
       ▼
   PDF Upload
       │
       ▼
 Text Extraction
       │
       ▼
 Model Inference
       │
       ▼
 JSON Response
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/multi-task-legal-ai.git
cd multi-task-legal-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask application

```bash
python app.py
```

The API will be available locally at:

```text
http://127.0.0.1:5000
```

---

## 📦 Requirements

Core dependencies include:

```text
Python
PyTorch
Transformers
Hugging Face
Flask
NLP libraries
PDF processing libraries
```

For the exact versions, refer to `requirements.txt`.

---

## 🎯 Project Objectives

The project aims to demonstrate how modern NLP and deep learning techniques can be applied to legal document analysis.

Key objectives include:

* Automating repetitive legal document analysis tasks.
* Applying BERT-based Transformer models to domain-specific text.
* Building a multi-task NLP pipeline.
* Serving trained models through a Flask backend.
* Creating a scalable architecture for multiple legal AI tasks.

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes**.

The predictions, summaries, extracted entities, and risk assessments generated by the system should **not be treated as legal advice or as a substitute for professional legal judgment**. Model outputs may contain errors or reflect limitations in the underlying training data and models.

---

## 🔮 Future Improvements

Potential improvements include:

* Fine-tuning BERT on larger legal-domain datasets.
* Integrating specialized Legal-BERT models.
* Improving long-document handling.
* Adding explainability for model predictions.
* Supporting additional legal NLP tasks.
* Adding confidence scores and evaluation metrics.
* Building a dedicated web interface.
* Implementing document-level semantic search.
* Adding model monitoring and evaluation pipelines.

---

## 👩‍💻 Author

**Linitta Elezabath Jimmy**

B.Tech Computer Science & Engineering — Artificial Intelligence & Machine Learning

Interested in **AI/ML, NLP, Full-Stack Development, and Intelligent Applications**.

---

## 📜 License

This project is intended for educational and research purposes. Add an appropriate open-source license if you plan to distribute the project publicly.
