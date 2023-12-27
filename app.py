from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import torch
from ray import serve
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
import minio

# MinIO Client setup
minio_client = minio.Minio(
    "10.223.2.3:9001",
    access_key="mlopskey",
    secret_key="mlopssecret",
    secure=False
)
def download_files_from_minio(bucket_name, file_names, save_path):
    for file_name in file_names:
        minio_client.fget_object(bucket_name, file_name, save_path + "/" + file_name)


class ModelConfigurator:
    def __init__(self):
        self.peft_starcoder_chat_save_pretrained = "./adapters"
        self.device = "cuda:0" 
        download_files_from_minio("mantis-shrimp-models", 
                                  ["adapter_config.json", "adapter_model.bin"], 
                                  self.peft_starcoder_chat_save_pretrained)
        self.config = PeftConfig.from_pretrained(self.peft_starcoder_chat_save_pretrained)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.setup_model()

    def setup_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name_or_path,
            return_dict=True,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path, add_eos_token=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = PeftModel.from_pretrained(self.model, self.peft_starcoder_chat_save_pretrained)

    async def generate(self, prompt: str):
        prompt_template = "<java_question>\n{query}\n<java_answer>"
        prompt = prompt_template.format(query=prompt)
        model_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.tokenizer.decode(
                self.model.generate(**model_input, max_new_tokens=256)[0],
                skip_special_tokens=True,
                eos_token_id=self.tokenizer.eos_token
            )
            return output

@serve.deployment
class PlateauGenerationModel:
    def __init__(self):
        self.model_configurator = ModelConfigurator()

    async def __call__(self, prompt: str = Form(...)):
        return await self.model_configurator.generate(prompt)

app = FastAPI()

@serve.deployment(route_prefix="/")
@serve.ingress(app)
class APIIngress:
    def __init__(self, model_handle):
        self.model_handle = model_handle

    @app.post("/plateau-generation")
    async def generate_plateau(self, prompt: str = Form(...)):
        return await self.model_handle.remote(prompt)
 

# Ray Serve'ı başlat
serve.start(detached=True)

# Deploymentları başlat
plateau_generation_model = PlateauGenerationModel.bind()
api_ingress = APIIngress.bind(plateau_generation_model)

# Define entrypoint
app = api_ingress
