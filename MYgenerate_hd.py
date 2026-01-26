import os
import json
import argparse
import inspect
from typing import Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def safe_get(item: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        v = item.get(k, None)
        if v is not None:
            return str(v)
    return default


def find_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for _, dev in model.hf_device_map.items():
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
    return next(model.parameters()).device


def build_prompt_text(system: str, user: str) -> str:
    system = (system or "").strip()
    user = (user or "").strip()
    
    # Llama-2 Chat Prompt Template
    return (
        f"<s>[INST] <<SYS>>\n"
        f"{system}\n"
        f"<</SYS>>\n\n"
        f"{user} [/INST]"
    )


def tokenize_prompt_and_answer(
    tokenizer,
    prompt_text: str,
    answer_text: str,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    answer_text = (answer_text or "").strip()

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    ans_ids = tokenizer(answer_text, add_special_tokens=False).input_ids

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("</s>")

    full_ids = prompt_ids + ans_ids + ([eos_id] if eos_id is not None else [])
    ans_start = len(prompt_ids)
    ans_end = len(prompt_ids) + len(ans_ids)

    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        full_ids = full_ids[overflow:]
        ans_start = max(0, ans_start - overflow)
        ans_end = max(0, ans_end - overflow)

    input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    return input_ids, attention_mask, ans_start, ans_end


@torch.inference_mode()
def extract_hd_hd3_from_answer(
    model,
    tokenizer,
    prompt_text: str,
    answer_text: str,
    max_length: int,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    input_ids, attention_mask, ans_start, ans_end = tokenize_prompt_and_answer(
        tokenizer, prompt_text, answer_text, max_length
    )

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Note: Accessing model.model for LlamaForCausalLM to get hidden states
    if hasattr(model, "model"):
        out = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True, output_hidden_states=False)
        
    hidden = out.last_hidden_state

    if ans_start > 0:
        # i_last (hd) is the internal vector of the last token of the input prompt
        last_vec = hidden[:, ans_start - 1, :].squeeze(0)
    else:
        # Fallback to last token if prompt is empty (shouldn't happen with standard templates)
        last_vec = hidden[:, -1, :].squeeze(0)

    if ans_end > ans_start:
        # i_resp (hd3) is the mean pooling of the internal vectors of the response tokens
        ans_hidden = hidden[:, ans_start:ans_end, :]
        mean_vec = ans_hidden.mean(dim=1).squeeze(0)
    else:
        # Fallback if no response tokens
        mean_vec = hidden.mean(dim=1).squeeze(0)

    hd = last_vec.detach().float().cpu().tolist()
    hd3 = mean_vec.detach().float().cpu().tolist()
    return hd, hd3


def main():
    parser = argparse.ArgumentParser(description="Extract Hidden States (hd, hd3) from Llama-like models.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM model directory.")
    parser.add_argument("--input_files", type=str, nargs="+", required=True, help="List of input JSON files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results with _hd.json suffix.")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f">>> Using GPU: {args.gpu}")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(">>> Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    ).to(device)

    model.eval()

    for file_path in args.input_files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
            
        print(f">>> Processing: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print(f"Error: {file_path} is not a JSON list. Skipping.")
            continue

        out_records = []
        pbar = tqdm(data, desc=os.path.basename(file_path))

        for item in pbar:
            system = safe_get(item, ["system"])
            user = safe_get(item, ["user"])
            answer = safe_get(item, ["model_response"])

            prompt_text = build_prompt_text(system, user)

            try:
                hd, hd3 = extract_hd_hd3_from_answer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    answer_text=answer,
                    max_length=args.max_length,
                    device=device,
                )
            except Exception as e:
                print(f"Error processing item in {file_path}: {e}")
                continue

            new_item = item.copy()
            new_item["hd"] = hd
            new_item["hd3"] = hd3
            out_records.append(new_item)

        out_name = os.path.basename(file_path).replace(".json", "_hd.json")
        out_path = os.path.join(args.output_dir, out_name)
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_records, f, ensure_ascii=False, indent=2)
            
        print(f">>> Saved: {out_path}")

    print(">>> All extraction tasks finished.")

if __name__ == "__main__":
    main()
