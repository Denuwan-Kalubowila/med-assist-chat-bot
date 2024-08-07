FROM python:3.12-alpine

WORKDIR /med-chat

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Debugging step: List the contents of the directory to ensure requirements.txt is present
RUN ls -la /med-chat/

# Install Python dependencies
RUN pip install -r /med-chat/requirements.txt

# Set environment variables from the .env file
ENV $(cat /med-chat/.env | xargs)

EXPOSE 80

CMD ["uvicorn", "med-chat.src.main:app", "--host", "0.0.0.0", "--port", "80"]