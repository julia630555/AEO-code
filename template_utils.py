def get_model_template(model_path, system, user, response=""):
    """
    根据模型路径识别模型类型并生成对应的 Prompt 模板。
    """
    path_lower = model_path.lower()
    
    # 1. Qwen 系列 (ChatML)
    if "qwen" in path_lower:
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        if response:
            prompt += response
        return prompt
        
    # 2. Phi-3 系列
    if "phi3" in path_lower or "phi-3" in path_lower:
        prompt = f"<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"
        if response:
            prompt += response
        return prompt
        
    # 3. Mistral 系列 (通常不带独立 SYS 标签，需拼入 USER)
    if "mistral" in path_lower:
        prompt = f"<s>[INST] {system}\n\n{user} [/INST]"
        if response:
            prompt += " " + response
        return prompt
        
    # 4. 默认 Llama-2 格式
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
    if response:
        prompt += " " + response
    return prompt
