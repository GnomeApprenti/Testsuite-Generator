import os
import base64
import subprocess
import traceback

from dotenv import load_dotenv
from huggingface_hub import login
from smolagents import CodeAgent, HfApiModel, Tool, OpenAIServerModel, TransformersModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from coverage import Coverage

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pedagogia_code_questions.RepoKnowledgeGraph import RepoKnowledgeGraph

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_AUTH = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)



#
# TOOLS
#

class FileTraverserTool(Tool):
    name = "file_traverser"
    description = "Parcourt un répertoire et retourne la liste des fichiers Python (.py) trouvés."
    inputs = {
        "directory": {"type": "string", "description": "Le chemin du répertoire à parcourir."}
    }
    output_type = "string"

    def forward(self, directory: str) -> str:
        python_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files

class CodeReaderTool(Tool):
    name = "code_reader"
    description = "Lit un fichier et retourne son contenu."
    inputs = {
        "filepath": {"type": "string", "description": "Le chemin du fichier à lire."}
    }
    output_type = "string"

    def forward(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

# class TestGeneratorTool(Tool):
#     name = "test_generator"
#     description = "Génère du code de tests pour un module Python donné à partir du contenu du code."
#     inputs = {
#         "code": {"type": "string", "description": "Le contenu du code Python pour lequel générer des tests."},
#         "module_name": {"type": "string", "description": "Le nom du module (sans l'extension .py)."}
#     }
#     output_type = "string"

#     def forward(self, code: str, module_name: str) -> str:
#         test_code = f'''import unittest
# import {module_name}

# class Test{module_name.capitalize()}(unittest.TestCase):
#     def test_placeholder(self):
#         self.assertTrue(True)

# if __name__ == '__main__':
#     unittest.main()
# '''
#         return test_code

class TestFileWriterTool(Tool):
    name = "test_file_writer"
    description = "Écrit le code de tests généré dans un fichier .py."
    inputs = {
        "test_code": {"type": "string", "description": "Le code des tests à écrire."},
        "output_path": {"type": "string", "description": "Le chemin du fichier de tests à créer"}
    }
    output_type = "string"

    def forward(self, test_code: str, output_path: str) -> str:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(test_code)
        return f"Le fichier de tests a été écrit dans {output_path}"

class CoverageCalculatorTool(Tool):
    name = "coverage_calculator"
    description = "Calcule la couverture de code avec coverage.py et génère un rapport HTML."
    inputs = {
        "tests_directory": {"type": "string", "description": "Le répertoire contenant les tests."},
        "source_directory": {"type": "string", "description": "Le répertoire des sources à analyser."}
    }
    output_type = "string"

    def forward(self, tests_directory: str, source_directory: str) -> str:
        cov = Coverage(source=[source_directory], branch=True)
        cov.start()
        result = subprocess.run(
            ["pytest", tests_directory, "--maxfail=1", "--disable-warnings"],
            capture_output=True,
            text=True
        )
        cov.stop()
        cov.save()
        report_output = subprocess.run(["coverage", "report", "-m"], capture_output=True, text=True)
        cov.html_report(directory="htmlcov")
        lines = report_output.stdout.splitlines()
        total_line = next((l for l in lines if l.strip().startswith("TOTAL")), "")
        return f"{total_line}\nRapport HTML : ./htmlcov/index.html"

#
# GENERATOR
#

def generate(path):
    #model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
    model = OpenAIServerModel(
        model_id="local-model",
        api_base="http://vllm:8000/v1",
        api_key="not-needed",
    )

    #tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True)
    #model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True)

    #transformers_model = TransformersModel(tokenizer=tokenizer, model=model)
    python_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    code = ""
    for file in python_files:
        with open(file) as f:
            code += f.read() + "\n"

    #knowledge graph
    graph = RepoKnowledgeGraph(path=path)
    graph.save_graph_to_file("graph.json")

    graph_str = ""
    with open('graph.json') as f:
        d = f.read()
        graph_str += d


    traverser = FileTraverserTool()
    reader = CodeReaderTool()
    #generator = TestGeneratorTool()
    writer = TestFileWriterTool()
    coverage_tool = CoverageCalculatorTool()
    tools = [traverser, reader]
    agent = CodeAgent(tools=tools, model=model, max_steps=10, additional_authorized_imports=['*', 'open', 'pytest', 'os', 'sys', 'subprocess', 'posixpath', 'unittest'])
    task = (
        "using the given tools, complete these steps : " + 
        "1. with the file_traverser tool, find all code files in the " + path + "/ directory\n" + 
        "2. with the code_reader tool, read the content of the files you found\n"
        "3. generate tests for the code you read\n"
        #"3. write the tests in a file and use it to calculate the coverage of the tests you just wrote "
        "4. return the tests you wrote. \n"
        "your final answer should be the code you generated, dont try to run the tests you generate just return them in the final answer\n" +
        "to help you understanding the structure of the code, you have the DAG representation of it as a json in this file : graph.json. use the code_reader_tool to read it, do NOT use open() \n"
    )
    output = agent.run(task)
    print(output)
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(output)
    return output

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TestsuiteRequest(BaseModel):
    path: str

#
# API ENDPOINTS
#

@app.post("/generate_testsuite")
async def run_testsuite_generator(request: TestsuiteRequest):
    try:
        print(f"Received project path: {request.path}")
        result = generate(request.path)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    return {"testsuite": result}

class CoverageRequest(BaseModel):
    tests_path: str
    source_path: str

@app.post("/calculate_coverage")
async def calculate_coverage(req: CoverageRequest):
    try:
        tool = CoverageCalculatorTool()
        result = tool.forward(req.tests_path, req.source_path)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    return {"coverage_report": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
