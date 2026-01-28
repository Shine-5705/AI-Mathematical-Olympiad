from typing import List
from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Abstract base for LLM generators."""

    @abstractmethod
    def get_candidates(self, context: str, n: int = 3) -> List[str]:
        """Generate n candidate continuations."""
        pass


class TransformersGenerator(BaseGenerator):
    """Generator using HuggingFace Transformers (works on any platform)."""

    def __init__(self, model_id: str = "AI-MO/NuminaMath-7B-TIR"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Install: pip install transformers torch")

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if torch.cuda.is_available():
            device_map = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            device_map = "mps"
            dtype = torch.float16
        else:
            device_map = "cpu"
            dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.device = device_map

    def get_candidates(self, context: str, n: int = 3) -> List[str]:
        """Generate multiple candidates."""
        import torch

        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        candidates = []

        for _ in range(n):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            candidates.append(response)

        return candidates


class MLXGenerator(BaseGenerator):
    """Generator using MLX for Apple Silicon (quantized models)."""

    def __init__(self, model_id: str = "mlx-community/Qwen2.5-Math-7B-Instruct-4bit"):
        try:
            from mlx_lm import load, generate
        except ImportError:
            raise ImportError("Install: pip install mlx-lm (Mac only)")

        self.model, self.tokenizer = load(model_id)
        self._generate = generate
        self.model_id = model_id

    def get_candidates(self, context: str, n: int = 3) -> List[str]:
        """Generate multiple candidates with temperature sampling."""
        candidates = []
        for _ in range(n):
            response = self._generate(
                self.model,
                self.tokenizer,
                prompt=context,
                max_tokens=512,
                temp=0.7,
            )
            candidates.append(response)
        return candidates


class VLLMGenerator(BaseGenerator):
    """Generator using vLLM for NVIDIA GPUs (fastest)."""

    def __init__(self, model_id: str = "AI-MO/NuminaMath-7B-TIR"):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Install: pip install vllm (requires NVIDIA GPU)")

        self.llm = LLM(model=model_id, trust_remote_code=True)
        self._SamplingParams = SamplingParams
        self.model_id = model_id

    def get_candidates(self, context: str, n: int = 3) -> List[str]:
        """Generate multiple candidates in parallel."""
        sampling_params = self._SamplingParams(
            n=n,
            temperature=0.7,
            max_tokens=512,
            stop=["\n\n", "###"]
        )
        outputs = self.llm.generate([context], sampling_params)
        return [o.text for o in outputs[0].outputs]


def create_generator(backend: str = "auto", **kwargs) -> BaseGenerator:
    """Factory function to create generator."""
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
    """Auto-detect the best available backend."""
    import platform

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
            try:
                import torch
                if torch.backends.mps.is_available():
                    return "transformers"
            except ImportError:
                pass

    return "transformers"
