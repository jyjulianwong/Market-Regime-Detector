from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from market_regime_detector.common.logging import configure_logging
from market_regime_detector.common.settings import get_settings
from market_regime_detector.server.routes import router

_settings = get_settings()
configure_logging(_settings.log_level)

app = FastAPI(title="Market Regime Detector API")

# Comma-separated exact origins, no wildcards (brief section 3.4) — empty by
# default, so no cross-origin access until a GitHub Pages origin is configured.
_allowed_origins = [
    origin.strip() for origin in _settings.cors_allowed_origins.split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(router)
