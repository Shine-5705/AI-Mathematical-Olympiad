import pandas as pd
from src.search import AIMO_MCTS


def main():
    df = pd.read_parquet("data/processed/math_reasoning_tir.parquet")
    problem = df.iloc[0]['instruction']

    engine = AIMO_MCTS()
    print("Starting MCTS Search...")
    best_solution = engine.search(problem, iterations=5)

    print("\n--- BEST VERIFIED SOLUTION ---")
    print(best_solution)


if __name__ == "__main__":
    main()
