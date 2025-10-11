import json
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from rag.retrieve import hybrid_search
from rag.rerank import rerank


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config
)


def generate_answer(query, context, max_new_tokens=200):    
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

    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "assistant" in answer.lower():
        answer = answer.split("assistant")[-1]
    return answer.strip()


def main():
    test_queries = [
        "Who is Nathan Davis?",
        "When was Carnegie Mellon University founded?",
        "What is CMU's yearbook called?"
    ]

    results = []

    for q in tqdm(test_queries, desc="Generating answers"):
        retrieved = hybrid_search(q, top_k=20)
        docs = [text for _, text, _ in retrieved]
        top_docs = rerank(q, docs, top_k=5)

        context = "\n".join(top_docs)

        answer = generate_answer(q, context)
        results.append({"query": q, "answer": answer})

        print(f"\n Q: {q}\n A: {answer}\n{'-'*60}")

    out_path = "data/generated_answers_qwen.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    print(f"\nSaved generated answers to {out_path}")


if __name__ == "__main__":
    main()
