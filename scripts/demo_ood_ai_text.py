import asyncio, random, time, os
import httpx
import certifi
from datasets import load_dataset
from scripts.demo_utils import BASE_URL, sustained_send, watch_for_breach_and_rollback
from dotenv import load_dotenv
from openai import AsyncOpenAI

print("Loading a small HC3 sample for realistic baseline traffic...")
_hc3 = load_dataset("json", data_files="data/hc3_all.jsonl", split="train")
_baseline_texts = []
for row in _hc3.select(range(200)):
    _baseline_texts += [a for a in row["human_answers"] if a.strip()]
    _baseline_texts += [a for a in row["chatgpt_answers"] if a.strip()]

def baseline_text() -> str:
    return random.choice(_baseline_texts)

load_dotenv()
CURRENT_LLM_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("CURRENT_LLM_API_KEY")

_FALLBACK_CURRENT_GEN_SAMPLES = [
    "That's a fair point, worth unpacking a little. The data suggests X, but "
    "there's real uncertainty about Y, so I'd hold that conclusion loosely "
    "rather than treat it as settled.",
    "Quick breakdown: the core issue is timing, not budget. The team's already "
    "flagged this as a risk, and I'd push back gently on the idea that more "
    "headcount fixes it on its own.",
    "Honestly, it depends. Optimizing for speed, go with the simpler approach. "
    "If correctness matters more here, the extra complexity is worth it.",
]

async def build_current_gen_pool(n: int = 30, delay_s: float = 2.1) -> list[str]:
    """Generates n samples sequentially, respecting Groq's 30 RPM limit (~1 per 2.1s)."""
    if not CURRENT_LLM_API_KEY:
        print("No API key found — pool will use default fallback samples.")
        return _FALLBACK_CURRENT_GEN_SAMPLES

    prompts = [
        "Write a short opinion on remote work.",
        "Give a brief take on whether AI will replace most jobs.",
        "Share a quick thought on the four-day work week.",
        "What's your view on social media's effect on attention spans?",
        "Briefly weigh in on whether college is still worth it.",
        "Explain how photosynthesis works in two sentences.",
        "Write a short product review for a pair of running shoes.",
        "Give step-by-step instructions for changing a bike tire.",
        "Write a brief, encouraging message to someone starting a new job.",
        "Summarize the plot of a mystery novel in three sentences.",
    ]
    pool = []
    http_client = httpx.AsyncClient(verify=certifi.where())
    client = AsyncOpenAI(
        api_key=CURRENT_LLM_API_KEY, 
        base_url="https://api.groq.com/openai/v1", 
        http_client=http_client
    )
    
    for i in range(n):
        try:
            resp = await client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": random.choice(prompts)}],
            )
            text = resp.choices[0].message.content
            if text:
                pool.append(text)
                print(f"  Generated sample {i+1}/{n}")
        except Exception as e:
            print(f"  [Pool generation error] {e} — skipping sample")
        await asyncio.sleep(delay_s)

    await http_client.aclose()
    return pool if pool else _FALLBACK_CURRENT_GEN_SAMPLES

async def main():
    async with httpx.AsyncClient(timeout=60.0) as client:
        async def send_baseline():
            await client.post(f"{BASE_URL}/predict/ai_text", json={"text": baseline_text()})

        print("=== AI-Text-Detection OOD Injection Demo ===")
        if not CURRENT_LLM_API_KEY:
            print("NOTE: No Groq/LLM API key found — using fallback samples.")

        print("\nPhase 1: Baseline traffic (real HC3 human + 2022-era ChatGPT text)")
        await sustained_send(send_baseline, duration_s=40, rate_per_s=5, label="ai_text-baseline")

        try:
            drift_resp = await client.get(f"{BASE_URL}/drift/ai_text", params={"version": "v2"}, timeout=30.0)
            print(f"\nBaseline drift: {drift_resp.text}")
        except httpx.ReadTimeout:
            print("\n[Warning] Drift calculation endpoint timed out — proceeding to Phase 2.")

        print("\nPre-generating current-gen LLM text pool (respecting API rate limits)...")
        current_gen_pool = await build_current_gen_pool(n=30)
        print(f"Pool ready: {len(current_gen_pool)} samples compiled.")

        async def send_shifted():
            await client.post(f"{BASE_URL}/predict/ai_text", json={"text": random.choice(current_gen_pool)})

        start_ts = time.time()
        print("\nPhase 2: Injecting current-generation LLM text (real generalization test)")
        await asyncio.gather(
            sustained_send(send_shifted, duration_s=90, rate_per_s=3, label="ai_text-injection"),
            watch_for_breach_and_rollback(client, "ai_text", "v2", start_ts, timeout_s=150),
        )

if __name__ == "__main__":
    asyncio.run(main())