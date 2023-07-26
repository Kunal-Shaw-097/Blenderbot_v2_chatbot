# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED = 1
ENV TRANSFORMERS_CACHE = "/root/cache/hf_cache_home"

WORKDIR /app

RUN python -m pip install --upgrade pip

# Copy the source code into the container.
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \ 
    python -m pip install -r requirements.txt 

# Expose the port that the application listens on.
EXPOSE 8501

# Run the application.
ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]
