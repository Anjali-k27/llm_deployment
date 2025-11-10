from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoOptimizationConfig
from pathlib import Path
import shutil
import os

MODEL_NAME = "microsoft/DialoGPT-small"
MODEL_DIR = Path("model")
QUANTIZED_DIR = Path("quantized_model")

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Save original model
tokenizer.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR)

print("Saving original model...")

# Export to ONNX (Open Neural Network Exchange)
onnx_model_path = MODEL_DIR / "model.onnx"
try:
    print("Exporting model to ONNX...")
    from optimum.exporters.onnx import main_export
    main_export(
        model=model,
        tokenizer=tokenizer,
        output=onnx_model_path,
        task="text-generation",
        opset=14
    )
except Exception as e:
    print("ONNX export failed:", str(e))
    print("Falling back to original PyTorch model.")
    exit()

# Quantize (Ex: 64 bit to 8 bit)
try:
    print("Creating quantized model...")
    quantizer = ORTQuantizer.from_pretrained(MODEL_DIR)
    quantizer.quantize(
        save_dir=QUANTIZED_DIR,
        optimization_config=AutoOptimizationConfig.avx512()  # CPU is compatible with 512 bit processing 
    )
except Exception as e:
    print("Quantization failed:", str(e))
    print("Falling back to using the original model.")
