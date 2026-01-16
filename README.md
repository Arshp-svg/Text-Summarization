# 📝 Text Summarization - End-to-End NLP Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.78.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)

An end-to-end deep learning project for automatic text summarization using transformer models (T5). The project features a complete MLOps pipeline with data ingestion, validation, transformation, model training, evaluation, and deployment using FastAPI and Streamlit.

---

## 🌟 Features

- **Transformer-Based Summarization**: Uses T5 model for state-of-the-art text summarization
- **Complete ML Pipeline**:  Modular pipeline architecture covering all stages of ML workflow
- **RESTful API**: FastAPI backend for model serving
- **Interactive UI**: Streamlit-based frontend for easy interaction
- **ROUGE Metrics**: Comprehensive evaluation using ROUGE scores
- **Configurable**:  YAML-based configuration for easy experimentation
- **Scalable Architecture**: Clean code structure following software engineering best practices

---

## 🏗️ Project Architecture

```
├── src/textSummaizer/
│   ├── components/          # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer. py
│   │   └── model_eval.py
│   ├── pipeline/            # Training and prediction pipelines
│   │   ├── stage01_data_ingestion.py
│   │   ├── stage02_data_validation.py
│   │   ├── stage03_data_transformation.py
│   │   ├── stage04_model_trainer.py
│   │   ├── stage05_mode_eval.py
│   │   └── prediction. py
│   ├── config/              # Configuration management
│   ├── entity/              # Data classes for configs
│   ├── constants/           # Project constants
│   ├── utils/               # Utility functions
│   └── logging/             # Custom logging setup
├── config/
│   └── config.yaml          # Main configuration file
├── params.yaml              # Model hyperparameters
├── app.py                   # FastAPI application
├── ui. py                    # Streamlit UI
├── main.py                  # Training pipeline execution
├── requirements.txt         # Python dependencies
└── research/                # Jupyter notebooks for experimentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Arshp-svg/Text-Summarization.git
   cd Text-Summarization
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Training the Model

Execute the complete training pipeline:

```bash
python main.py
```

This will run all stages: 
- **Stage 1**: Data Ingestion (downloads SAMSum dataset)
- **Stage 2**: Data Validation (validates dataset schema)
- **Stage 3**: Data Transformation (tokenizes and prepares data)
- **Stage 4**: Model Training (fine-tunes T5 model)
- **Stage 5**: Model Evaluation (computes ROUGE metrics)

### 2. Running the API Server

Start the FastAPI backend:

```bash
python app.py
```

The API will be available at `http://localhost:8080`

**API Endpoints**:
- `GET /`: Redirects to API documentation
- `GET /train`: Triggers model training
- `GET /predict? text=<your_text>`: Returns summary for input text

View interactive API docs at:  `http://localhost:8080/docs`

### 3. Running the Streamlit UI

In a new terminal window:

```bash
streamlit run ui.py
```

Access the UI at `http://localhost:8501`

---

## 📊 Model Details

- **Base Model**: T5 (Text-to-Text Transfer Transformer)
- **Task**: Abstractive Text Summarization
- **Dataset**: SAMSum (Dialogue Summarization)
- **Tokenizer**: T5 Tokenizer
- **Max Input Length**: 1024 tokens
- **Max Output Length**: 256 tokens
- **Evaluation Metrics**:  ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum

### Training Parameters

Configure hyperparameters in `params.yaml`:
- Learning rate
- Batch size
- Number of epochs
- Weight decay
- Gradient accumulation steps
- Beam search parameters

---

## 🔧 Configuration

### `config/config.yaml`
Defines paths and configurations for each pipeline stage: 
- Data ingestion settings
- Model paths
- Artifact directories
- Tokenizer configurations

### `params.yaml`
Contains model hyperparameters and training settings. 

---

## 📁 Dataset

The project uses the **SAMSum Corpus** - a dataset containing messenger-like conversations with abstractive summaries. 

- **Training samples**: ~14,700
- **Validation samples**: ~800
- **Test samples**: ~800

---

## 🎯 Development Workflow

When making changes to the project, follow this workflow:

1. Update `config/config.yaml` (if adding new paths/configs)
2. Update `params.yaml` (if changing hyperparameters)
3. Update entity classes in `src/textSummaizer/entity/`
4. Update configuration manager in `src/textSummaizer/config/configuration.py`
5. Update components in `src/textSummaizer/components/`
6. Update pipeline stages in `src/textSummaizer/pipeline/`
7. Update `main.py` (if adding new stages)
8. Update `app.py` (if adding new API endpoints)

---

## 📈 Model Evaluation

The model is evaluated using ROUGE metrics: 

- **ROUGE-1**:  Unigram overlap
- **ROUGE-2**:  Bigram overlap  
- **ROUGE-L**: Longest common subsequence
- **ROUGE-Lsum**: Summary-level ROUGE-L

Results are saved to `artifacts/model_evaluation/metrics. csv`

---

## 🐳 Docker Support

Build and run using Docker:

```bash
docker build -t text-summarizer .
docker run -p 8080:8080 text-summarizer
```

---

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Web Framework**: FastAPI
- **Frontend**: Streamlit
- **Data Processing**: Pandas, Datasets
- **Evaluation**: ROUGE Score, SacreBLEU
- **Configuration**: PyYAML, python-box
- **Logging**: Custom logging module

---

## 📝 Project Structure Highlights

### Modular Components
- **Data Ingestion**: Downloads and extracts SAMSum dataset
- **Data Validation**: Validates required files and schema
- **Data Transformation**: Tokenizes text and prepares training data
- **Model Trainer**: Fine-tunes T5 model with specified parameters
- **Model Evaluation**:  Computes ROUGE metrics on test set

### Pipeline Architecture
Each stage is encapsulated in a pipeline class for easy execution and maintenance.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

**Author**: Arshp-svg  
**Email**: arshpatel213@gmail.com  
**GitHub**: [@Arshp-svg](https://github.com/Arshp-svg)

---

## 📄 License

This project is open-source and available under the MIT License. 

---

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- SAMSum dataset creators
- FastAPI and Streamlit communities

---

## 🔮 Future Enhancements

- [ ] Add support for multiple summarization models
- [ ] Implement document summarization (longer texts)
- [ ] Add multilingual support
- [ ] Deploy to cloud platforms (AWS/GCP/Azure)
- [ ] Add user authentication
- [ ] Implement model versioning and A/B testing
- [ ] Add support for custom datasets

---

**⭐ If you find this project helpful, please give it a star! **
