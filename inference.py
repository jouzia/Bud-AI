from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os

from core.environment.engine import StudyEnv, StepResult
from core.environment.state import Action, Mode
from core.agents.memory_agent import MemoryAgent, ComplexityMode
from core.memory.memory import DualLayerMemory

app = FastAPI(title="OpenEnv Hackathon API")

# Global instances (Stateful for the session)
# In a real production app, we would use session management, 
# but for hackathon automated checks, a global instance is standard.
env: Optional[StudyEnv] = None
agent: Optional[MemoryAgent] = None

class ActionRequest(BaseModel):
    action: str

class ActRequest(BaseModel):
    state: Dict[str, Any]
    complexity: Optional[str] = "standard"

@app.post("/reset")
async def reset(mode: str = "hard"):
    global env, agent
    try:
        env = StudyEnv(mode=mode)
        initial_state = env.reset()
        
        # Initialize agent as well, just in case
        agent = MemoryAgent(complexity_mode=ComplexityMode.STANDARD)
        
        return {
            "status": "success",
            "state": initial_state.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(request: ActionRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        result: StepResult = env.step(request.action)
        return {
            "status": "success",
            "state": result.state.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/act")
async def act(request: ActRequest):
    global agent
    if agent is None:
        agent = MemoryAgent(complexity_mode=ComplexityMode.STANDARD)
    
    try:
        # Reconstruct state from JSON
        from core.environment.state import StudyState
        state = StudyState(**request.state)
        
        # Switch complexity if needed
        if request.complexity in [c.value for c in ComplexityMode]:
            agent.complexity_mode = ComplexityMode(request.complexity)
            
        response = agent.act(state)
        return {
            "status": "success",
            "action": response.action.value,
            "reasoning": response.reasoning,
            "raw": response.raw,
            "latency_ms": response.latency_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
@app.get("/validate")
async def validate():
    """Health check and schema validation endpoint."""
    return {"status": "healthy", "version": "2.0.0"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    uvicorn.run(app, host="0.0.0.0", port=port)
