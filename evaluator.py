import pandas as pd
import asyncio
import json
import os
from sentence_transformers import SentenceTransformer
import asyncio

# Load your retrieval and generation logic from cb.py
from cb import retrieve_relevant_chunks, generate_answer_async, query_llm_async

async def evaluate_rag():
    """
    Run evaluation on all questions in eval_questions1.csv
    and use the configured LLM (Groq or Ollama) to judge results.
    """
    # 1. Load test data
    if not os.path.exists("eval_questions1.csv"):
        print("✗ eval_questions1.csv not found.")
        return

    df = pd.read_csv("eval_questions1.csv")
    results = []

    print(f"🚀 Starting evaluation of {len(df)} questions...")

    for idx, row in df.iterrows():
        question = row["question"]
        expected = row["expected_answer"]

        print(f"\n[{idx+1}/{len(df)}] Query: {question}")

        # 2. RAG Pipeline: Retrieve + Generate
        retrieved = retrieve_relevant_chunks(question, top_k=3)
        response = await generate_answer_async(question, retrieved)

        generated_answer = response["answer"]
        sources = [s["document"] for s in response["sources"]]

        # 3. LLM-as-a-Judge: Score the result
        # We ask the LLM to compare generated vs expected based only on the context provided
        judge_prompt = f"""
        You are an impartial evaluator for a RAG system.
        Evaluate the 'Generated Answer' based on the 'Expected Answer' and the 'User Question'.

        QUESTION: {question}
        EXPECTED ANSWER: {expected}
        GENERATED ANSWER: {generated_answer}

        Provide three scores from 0.0 to 1.0:
        1. RELEVANCE: How well does the answer address the question?
        2. FAITHFULNESS: Is the answer supported by fact, or does it hallucinate?
        3. ACCURACY: How closely does the generated answer match the meaning of the expected answer?

        Output ONLY a JSON object like this:
        {{"relevance": 0.9, "faithfulness": 1.0, "accuracy": 0.8, "reasoning": "short explanation"}}
        """

        try:
            # Reuse the LLM configured in cb.py to act as judge
            judge_json = await query_llm_async(judge_prompt)
            # Basic cleaning in case LLM adds markdown wrappers
            judge_json = judge_json.replace("```json", "").replace("```", "").strip()
            scores = json.loads(judge_json)
        except Exception as e:
            print(f"  ⚠️ Judging failed: {e}")
            scores = {"relevance": 0, "faithfulness": 0, "accuracy": 0, "reasoning": str(e)}

        print(f"  Result: Rel={scores['relevance']} Faith={scores['faithfulness']} Acc={scores['accuracy']}")

        results.append({
            "id": idx + 1,
            "question": question,
            "expected_answer": expected,
            "generated_answer": generated_answer,
            "sources": sources,
            "scores": scores
        })

        # Small sleep to respect Groq rate limits if needed
        await asyncio.sleep(1)

    # 4. Save and Report
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Calculate averages
    avg_rel = sum(r["scores"]["relevance"] for r in results) / len(results)
    avg_fai = sum(r["scores"]["faithfulness"] for r in results) / len(results)
    avg_acc = sum(r["scores"]["accuracy"] for r in results) / len(results)

    print("\n" + "="*30)
    print("📊 EVALUATION SUMMARY")
    print("="*30)
    print(f"Avg Relevance:   {avg_rel:.2f}")
    print(f"Avg Faithfulness: {avg_fai:.2f}")
    print(f"Avg Accuracy:     {avg_acc:.2f}")
    print(f"Total Tested:     {len(results)}")
    print("="*30)
    print("Full details saved to eval_results.json")

if __name__ == "__main__":
    asyncio.run(evaluate_rag())