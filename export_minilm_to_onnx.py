from transformers import AutoTokenizer, AutoModel
import torch

model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Load model + tokenizer
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dummy_input = tokenizer("This is a test input.", return_tensors="pt")

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "minilm-sentence-transformer.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["sentence_embedding"],
    opset_version=14,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "sentence_embedding": {0: "batch_size"}
    }
)
print("ONNX model exported: minilm-sentence-transformer.onnx")
