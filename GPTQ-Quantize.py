from transformers import AutoTokenizer
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

def quantize_model():
    # Model and quantization paths
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    quant_path = "./Llama-3.2-3B-Instruct-GPTQ-4bit-32-256-filtering-1356"

    calibration_dataset = []  # List to store filtered calibration texts
    calibration_size = 256    # Number of samples for calibration
    min_token_len = 256       # Minimum token length for a sample
    max_token_len = 2048      # Maximum token length for a sample
    batch_size = 8              # Batch size for quantization
    group_size = 32            # Group size for quantization
    bits = 4                   # Number of bits for quantization


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load dataset
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train"
    )


    # Filter dataset for calibration
    for text in dataset["text"]:
        if not text.strip():
            continue

        tokenized = tokenizer(text, truncation=True, max_length=max_token_len)
        if len(tokenized["input_ids"]) >= min_token_len:
            calibration_dataset.append(text.strip())

        if len(calibration_dataset) >= calibration_size:
            break 

    # Set quantization configuration
    quant_config = QuantizeConfig(bits=bits, group_size=group_size)

    # Load model with quantization config
    model = GPTQModel.load(model_name, quant_config)

    # Quantize the model using the calibration dataset
    model.quantize(calibration_dataset, batch_size=batch_size)

    # Save the quantized model
    model.save(quant_path)

    print(f"Quantized model saved to {quant_path}")

if __name__ == "__main__":
    quantize_model()