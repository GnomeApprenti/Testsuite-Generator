FROM python:3.11-slim

WORKDIR /app

RUN pip install smolagents
RUN pip install uvicorn
RUN pip install fastapi
RUN pip install huggingface_hub
RUN pip install opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents
RUN pip install langfuse
RUN pip install smolagents[openai]
RUN pip install coverage
RUN pip install pytest
RUN pip install smolagents[transformers]
RUN pip install hf_xet
RUN pip install matplotlib
RUN pip install tenacity
RUN pip install langchain_text_splitters
RUN pip install chromadb
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

