from typing import List
from vllm import LLM, SamplingParams


class NuminaGenerator:
    """High-throughput candidate generation using vLLM with NuminaMath."""

    def __init__(self, model_id: str = "AI-MO/NuminaMath-7B-TIR"):
        self.llm = LLM(model=model_id, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            n=3,
            temperature=0.7,
            max_tokens=512,
            stop=["\n\n", "###"]
        )

    def get_candidates(self, context: str) -> List[str]:
        """Generate multiple candidate continuations for the given context."""
        outputs = self.llm.generate([context], self.sampling_params)
        return [o.text for o in outputs[0].outputs]
