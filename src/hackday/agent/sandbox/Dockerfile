FROM python:3.12-slim

# `jq` is recommended in the system prompt for reading individual problems
# from /work/problems.jsonl, so make sure it's available.
RUN apt-get update \
    && apt-get install -y --no-install-recommends jq \
    && rm -rf /var/lib/apt/lists/*

# Inspect's bash/python tools work out of the box. Add math conveniences.
RUN pip install --no-cache-dir sympy numpy

# Working directory the agent operates in.
WORKDIR /work

ENV PYTHONUNBUFFERED=1
