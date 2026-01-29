"""LLM generators for MCTS candidate generation."""
import platform
from typing import List
from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    @abstractmethod
    def get_candidates(self, context: str, n: int = 3) -> List[str]:
        pass


class MLXGenerator(BaseGenerator):
    """MLX generator — supports LoRA adapters natively."""

    def __init__(self, model_id: str = "mlx-community/Qwen2.5-Math-7B-Instruct-4bit", adapter_path: str = None):
        from mlx_lm import load, generate

        if adapter_path:
            self.model, self.tokenizer = load(model_id, adapter_path=adapter_path)
        else:
            self.model, self.tokenizer = load(model_id)

        self._generate = generate
        self.model_id = model_id

    def get_candidates(self, context: str, n: int = 3) -> List[str]:
        candidates = []
        for _ in range(n):
            response = self._generate(
                self.model, self.tokenizer,
                prompt=context,
                max_tokens=512,
                temp=0.7,
            )
            candidates.append(response)
        return candidates


class TransformersGenerator(BaseGenerator):
    """Transformers generator — uses 4-bit on CUDA, supports LoRA adapters."""

    def __init__(self, model_id: str = "AI-MO/NuminaMath-7B-TIR", adapter_path: str = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                ),
                device_map="auto", trust_remote_code=True,
            )
            self.device = "cuda"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32,
                device_map="cpu", trust_remote_code=True,
            )
            self.device = "cpu"

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        self.model_id = model_id

    def get_candidates(self, context: str, n: int = 3) -> List[str]:
        import torch

        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        candidates = []
        for _ in range(n):
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            candidates.append(text)
        return candidates


class VLLMGenerator(BaseGenerator):
    """vLLM generator — fastest on NVIDIA GPUs, batch generation."""

    def __init__(self, model_id: str = "AI-MO/NuminaMath-7B-TIR"):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_id, trust_remote_code=True)
        self._SamplingParams = SamplingParams
        self.model_id = model_id

    def get_candidates(self, context: str, n: int = 3) -> List[str]:
        params = self._SamplingParams(n=n, temperature=0.7, max_tokens=512)
        outputs = self.llm.generate([context], params)
        return [o.text for o in outputs[0].outputs]


def create_generator(backend: str = "auto", **kwargs) -> BaseGenerator:
    if backend == "auto":
        backend = _detect_backend()

    generators = {
        "transformers": TransformersGenerator,
        "mlx": MLXGenerator,
        "vllm": VLLMGenerator,
    }

    if backend not in generators:
        raise ValueError(f"Unknown backend: {backend}. Choose from {list(generators.keys())}")
    return generators[backend](**kwargs)


def _detect_backend() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            try:
                import vllm
                return "vllm"
            except ImportError:
                return "transformers"
    except ImportError:
        pass

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_lm
            return "mlx"
        except ImportError:
            pass

    return "transformers"
