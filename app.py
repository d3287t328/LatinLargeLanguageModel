import os
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

app = Flask(__name__)

model_path = os.environ.get("MODEL_PATH", "your_output_directory")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "your_output_directory")

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model.config.decoder_start_token_id = tokenizer.eos_token_id

@app.route("/generate", methods=["POST"])
def generate():
    input_text = request.json.get("input_text", "")
    max_length = request.json.get("max_length", 128)
    temperature = request.json.get("temperature", 0.7)
    top_k = request.json.get("top_k", 50)
    top_p = request.json.get("top_p", 0.95)

    inputs = tokenizer.encode_plus(input_text.strip(), return_tensors="pt", max_length=max_length, truncation=True)
    output_tokens = model.generate(**inputs, max_new_tokens=1000, temperature=temperature, top_k=top_k, top_p=top_p)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True).replace("Ġ", " ")

    return jsonify({"generated_text": output_text.strip()})

if __name__ == "__main__":
    app.run()
