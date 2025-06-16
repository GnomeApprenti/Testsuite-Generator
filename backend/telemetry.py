from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

class TelemetrySetup:
    """Setup OpenTelemetry tracing for smolagents."""
    
    def __init__(self):
        self.trace_provider = None
    
    def setup_tracing(self):
        """Configure and start OpenTelemetry tracing."""
        self.trace_provider = TracerProvider()
        self.trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
        SmolagentsInstrumentor().instrument(tracer_provider=self.trace_provider)
        return self.trace_provider

# Global telemetry instance
telemetry = TelemetrySetup()