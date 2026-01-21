FROM python:3.11-slim

WORKDIR /app

# System deps (optional minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY app.py /app/app.py
COPY docs /app/docs
COPY README.md /app/README.md
COPY README_UNIFIED_SUITE.md /app/README_UNIFIED_SUITE.md
COPY CANON_MAP.md /app/CANON_MAP.md

RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install -e .

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
