# rag_src/generate_answers.py
import json
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from rag_src.retrieve import hybrid_search
from rag_src.rerank import rerank

# ============================================================
#  Model Setup (Qwen2.5-3B-Instruct)
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Optional 4-bit quantization for faster inference on Kaggle T4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"🔹 Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config
)

# ============================================================
#  Helper: Generate Answer
# ============================================================
def generate_answer(query, context, max_new_tokens=200):
    """Generate an answer using Qwen2.5-3B-Instruct with proper chat template."""
    messages = [
        {"role": "system", "content": "You are a concise and factual assistant."},
        {
            "role": "user",
            "content": f"Answer the following question using only the context provided.\n\nContext:\n{context}\n\nQuestion: {query}",
        },
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=False,
        )

    # Decode and clean
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Cut off the prompt part
    if "assistant" in answer.lower():
        answer = answer.split("assistant")[-1]
    return answer.strip()


# ============================================================
#  Main Pipeline
# ============================================================
def main():
    test_queries = [
        "Who is Nathan Davis?",
        "When was Carnegie Mellon University founded?",
        "What is CMU's yearbook called?"
    ]

    results = []

    for q in tqdm(test_queries, desc="🧠 Generating answers"):
        # Step 1: Retrieve and rerank top chunks
        retrieved = hybrid_search(q, top_k=20)
        docs = [text for _, text, _ in retrieved]
        top_docs = rerank(q, docs, top_k=5)

        # Step 2: Join context
        context = "\n".join(top_docs)

        # Step 3: Generate answer
        answer = generate_answer(q, context)
        results.append({"query": q, "answer": answer})

        print(f"\n🔹 Q: {q}\n💬 A: {answer}\n{'-'*60}")

    # Step 4: Save outputs
    out_path = "data/generated_answers_qwen.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    print(f"\n✅ Saved generated answers to {out_path}")

# ============================================================
if __name__ == "__main__":
    main()
