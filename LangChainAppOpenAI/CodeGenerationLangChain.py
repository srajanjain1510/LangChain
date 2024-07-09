import os

import torch
import transformers
from transformers import AutoTokenizer

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'


def code_generation_model_impl():
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    # tokenizer.encode_plus(truncation=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model="codellama/CodeLlama-7b-hf",
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    sequences = pipeline(
        'def fibonacci(',
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1000

    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


code_generation_model_impl()
