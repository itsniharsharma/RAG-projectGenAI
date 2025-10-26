import os
from flask import Flask, render_template, request, jsonify
from rag_utils import query_rag  
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# os.environ["HF_HOME"] = "D:\\huggingface_cache"
# os.environ["TRANSFORMERS_CACHE"] = "D:\\huggingface_cache"

app = Flask(__name__)


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    device_map="auto" 
)

def generate_answer(query):
    context = query_rag(query, top_k=5) 
    prompt = f"Answer the user query based on the astrology books:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,  
        temperature=0.7,     
        top_p=0.9,
        do_sample=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    answer = generate_answer(data['query'])
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
