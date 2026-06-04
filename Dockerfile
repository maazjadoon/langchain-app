FROM python:3.10-slim

WORKDIR /app

# Copy the requirements file and install dependencies
# We do this before copying the rest of the app to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the application runs on
EXPOSE 5000

# Set default environment variables
# Note: For accessing Ollama on the host machine from inside Docker, 
# you often need to use host.docker.internal instead of localhost.
ENV OLLAMA_LLM_MODEL=llama3.2
ENV OLLAMA_EMBED_MODEL=nomic-embed-text
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

# Command to run the application
CMD ["python", "app.py"]
