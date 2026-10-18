from mangum import Mangum

from market_regime_detector.server.app import app

# Brief section 3.4: "FastAPI app wrapped with Mangum for Lambda." This is what a
# deployed Lambda points at (`main.handler`, Phase 8); local `uvicorn` points at
# `server.app:app` directly instead.
handler = Mangum(app)
