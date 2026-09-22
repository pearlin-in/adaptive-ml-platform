import asyncio, random, time, os
import httpx
from datasets import load_dataset
from scripts.demo_utils import BASE_URL, sustained_send, watch_for_breach_and_rollback
from dotenv import load_dotenv
from openai import AsyncOpenAI

print("Loading a small HC3 sample for realistic baseline traffic...")

_hc3 = load_dataset(
    "json",
    data_files="https://huggingface.co/datasets/hello-simpleai/hc3/resolve/main/all.jsonl",
    split="train",
)
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

async def generate_current_llm_text() -> str:
    if not CURRENT_LLM_API_KEY:
        print("[Warning] No API key found in .env, using fallback sample.")
        return random.choice(_FALLBACK_CURRENT_GEN_SAMPLES)
    try:
        client = AsyncOpenAI(
            api_key=CURRENT_LLM_API_KEY, 
            base_url="https://api.groq.com/openai/v1"
        )
        resp = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Write a short opinion on remote work."}],
        )
        return resp.choices[0].message.content or random.choice(_FALLBACK_CURRENT_GEN_SAMPLES)
    except Exception as e:
        print(f"[Groq API Error] {e} — falling back to sample text.")
        return random.choice(_FALLBACK_CURRENT_GEN_SAMPLES)
    
async def main():
    async with httpx.AsyncClient(timeout=15.0) as client:
        async def send_baseline():
            await client.post(f"{BASE_URL}/predict/ai_text", json={"text": baseline_text()})
        async def send_shifted():
            await client.post(f"{BASE_URL}/predict/ai_text", json={"text": await generate_current_llm_text()})

        print("=== AI-Text-Detection OOD Injection Demo ===")
        if not CURRENT_LLM_API_KEY:
            print("NOTE: CURRENT_LLM_API_KEY not set — using fallback samples, not a live "
                  "current-gen model. Set the env var and wire a provider above for the real test.")

        print("\nPhase 1: baseline traffic (real HC3 human + 2022-era ChatGPT text)")
        await sustained_send(send_baseline, duration_s=40, rate_per_s=5, label="ai_text-baseline")

        print(f"\nBaseline drift: {await client.get(f'{BASE_URL}/drift/ai_text', params={'version': 'v2'})}")

        start_ts = time.time()
        print("\nPhase 2: injecting current-generation LLM text (real generalization test)")
        await asyncio.gather(
            sustained_send(send_shifted, duration_s=90, rate_per_s=3, label="ai_text-injection"),
            watch_for_breach_and_rollback(client, "ai_text", "v2", start_ts, timeout_s=150),
        )

if __name__ == "__main__":
    asyncio.run(main())