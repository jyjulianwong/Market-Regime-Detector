FROM public.ecr.aws/lambda/python:3.12

# Built for linux/amd64 in CI (docker buildx build --platform linux/amd64) —
# matches established convention over the brief's arm64 ASSUMPTION.

COPY pyproject.toml uv.lock ./
COPY src ./src

# Installs this project itself plus the `batch` extra only — hmmlearn stays
# excluded (it's a `train`-extra dependency), keeping the batch Lambda's
# image free of the one dependency the brief says it must never import
# (brief section 3.5; prediction/ never imports training/).
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache ".[batch]"

CMD ["market_regime_detector.prediction.handler.handler"]
