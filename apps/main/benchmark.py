from omegaconf import OmegaConf
from dataclasses import dataclass, field
import time
import torch
from lingua.args import dataclass_from_dict

from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
    pack_prompts,
    sample_tokens,
)

# about 200 word prompt
PROMPT = """You are an assistant helping a researcher design an efficient and reliable long-form text generation benchmark. Write a detailed explanation of how to evaluate generation speed and throughput for a large language model under realistic serving conditions. Your answer should discuss why tokens-per-second depends on batch size, context length, attention complexity, and KV cache growth during decoding. Explain how prompt processing (“prefill”) differs from autoregressive decoding, and why measuring them separately is important. Include practical guidance on controlling for variability, such as using fixed sampling parameters, disabling streaming, warming up the model, and running multiple trials. Also describe how memory bandwidth and GPU utilization can become bottlenecks even when compute is available, especially at long context lengths. Finally, propose a minimal experimental protocol that reports average throughput, tail latency (p95), and total wall-clock time, and briefly mention how techniques like continuous batching, paged KV cache, and speculative decoding may change the results. Keep the writing clear and technical, and aim for a structured response that would be useful for an engineer optimizing inference performance."""

@dataclass
class BenchmarkConfig:
    warmup_iters: int = 3
    iters: int = 10
    batch_size: int = 1
    

def main():
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    gen_cfg = dataclass_from_dict(
        PackedCausalTransformerGeneratorArgs, cfg, strict=False
    )
    benchmark_cfg = dataclass_from_dict(
        BenchmarkConfig, cfg, strict=False
    )
    print(cfg)

    model, tokenizer, _ = load_consolidated_model_and_tokenizer(cfg.ckpt)

    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)
    
    # based on batch size, copy the prompt to the batch size
    prompts = [PROMPT] * benchmark_cfg.batch_size
    
    for _ in range(benchmark_cfg.warmup_iters):
        torch.cuda.synchronize()
        _ = generator.generate(prompts)
    torch.cuda.synchronize()
    
    e2e_tokens = 0
    t0 = time.perf_counter()
    for _ in range(benchmark_cfg.iters):
        generation, *_ = generator.generate(prompts)
        e2e_tokens += len(generation)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    e2e_s = t1 - t0
    
    print(f"E2E tokens: {e2e_tokens}")
    print(f"E2E time: {e2e_s}")
    print(f"E2E tokens/s: {e2e_tokens / e2e_s}")
    

if __name__ == "__main__":
    main()
    