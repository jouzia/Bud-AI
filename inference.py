import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from openai import OpenAI

from core.environment.engine import StudyEnv, StepResult
from core.environment.state import Action, Mode, StudyState

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & HACKATHON COMPLIANCE
# ─────────────────────────────────────────────────────────────────────────────

# Must follow the checklist: read from environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    # Per checklist: fail if HF_TOKEN is missing
    raise ValueError("HF_TOKEN must be provided in environment variables")

# Initialize OpenAI client as required
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

app = FastAPI(title="OpenEnv Hackathon API - Compliance Edition")

# Global state for the simulation
env: Optional[StudyEnv] = None
current_step = 0
total_reward = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURED LOGGING (START/STEP/END)
# ─────────────────────────────────────────────────────────────────────────────

def log_event(tag: str, content: str):
    """Prints structured logs to stdout for the automated evaluator."""
    print(f"[{tag}] {content}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# API MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ActionRequest(BaseModel):
    action: str

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/reset")
async def reset(mode: str = "hard"):
    global env, current_step, total_reward
    try:
        env = StudyEnv(mode=Mode(mode))
        initial_state = env.reset()
        current_step = 0
        total_reward = 0.0
        
        # Required structured log
        log_event("START", f"episode={datetime.utcnow().isoformat()} mode={mode}")
        
        return {
            "status": "success",
            "state": initial_state.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(request: ActionRequest):
    global env, current_step, total_reward
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step")
    
    try:
        current_step += 1
        result: StepResult = env.step(request.action)
        total_reward += result.reward
        
        # Required structured log
        log_event("STEP", f"num={current_step} action={request.action} reward={result.reward} total={total_reward}")
        
        if result.done:
            log_event("END", f"total_reward={total_reward} steps={current_step}")
            
        return {
            "status": "success",
            "state": result.state.model_dump(),
            "reward": result.reward,
            "done": result.done
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/act")
async def act(state: Dict[str, Any]):
    """
    Hackathon requirement: All LLM calls use the OpenAI client.
    This replaces the previous Gemini-based agent for the purpose of the submission.
    """
    try:
        # Prompt construction (simplified for the OpenAI interface)
        prompt = f"Given this OpenEnv StudyState: {json.dumps(state)}, what is the next best action? Choose from: expand, summarize, quiz, reorganize. Respond with ONLY the action name."
        
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a strategic Study Agent. Respond in JSON or plain text action."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        latency = round((time.perf_counter() - t0) * 1000, 1)
        
        raw_action = response.choices[0].message.content.strip().lower()
        
        # Basic validation to ensure we return a valid action
        valid_actions = ["expand", "summarize", "quiz", "reorganize", "do_nothing"]
        action = next((a for a in valid_actions if a in raw_action), "do_nothing")
        
        return {
            "status": "success",
            "action": action,
            "latency_ms": latency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/validate")
async def validate():
    return {"status": "healthy", "compliance": "full"}

if __name__ == "__main__":
    # Hugging Face Spaces usually environment variables for port
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
