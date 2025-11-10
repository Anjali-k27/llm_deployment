# Quantized LLM Chat Application

This repository contains a simple web application that uses a quantized and compressed Large Language Model (LLM) for a chat interface. The application is built with Streamlit and containerized with Docker.

## Features

- **Model Quantization**: Uses `optimum` to quantize a Hugging Face model (`microsoft/DialoGPT-small`) for more efficient inference.
- **Streamlit UI**: A clean and simple chat interface.
- **Dockerized**: Easy to deploy using Docker and Docker Compose.

## Project Structure

```
llm-quantized-app/
├── .git/
├── model/              # Stores the original downloaded model
├── quantized_model/    # Stores the quantized model
├── app.py              # The Streamlit application
├── quantize.py         # Script to download and quantize the model
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
└── README.md           # This file
```

## Getting Started

Follow these instructions to get the application up and running on your local machine.

### Prerequisites

- [Python 3.9+](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Installation & Setup

**1. Clone the Repository**

```bash
git clone <your-repository-url>
cd llm-quantized-app
```

**2. Quick Setup (Recommended)**

Run the automated setup script:

```bash
python setup.py
```

This will automatically install dependencies and download/quantize the model.

**3. Manual Setup (Alternative)**

If you prefer to set up manually:

**Install Dependencies**

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

**Quantize the Model**

Run the quantization script to download and prepare the model. This will create the `model` and `quantized_model` directories.

```bash
python quantize.py
```

This step might take some time as it downloads the model from Hugging Face.

### Running the Application

There are two ways to run the application:

**Option A: Using Docker (Recommended)**

This is the easiest way to get started.

```bash
docker-compose up --build
```

**Option B: Running Locally**

If you prefer to run the app without Docker:

```bash
streamlit run app.py
```

### Access the Application

Once the application is running, open your browser and navigate to:

[http://localhost:8501](http://localhost:8501)

## How It Works

1.  **`quantize.py`**: This script downloads the `microsoft/DialoGPT-small` model and its tokenizer from the Hugging Face Hub. It then uses `optimum` and its ONNX Runtime backend to convert the model to the ONNX format and apply quantization. The original and quantized models are saved locally.

2.  **`app.py`**: This is a Streamlit application that loads the quantized model from disk. It provides a user interface for chatting with the model. It caches the model resource to avoid reloading on every interaction.

3.  **`Dockerfile`**: Defines the container image for the application. It installs dependencies, copies the application code, and sets up the environment to run the quantization script followed by the Streamlit app.

