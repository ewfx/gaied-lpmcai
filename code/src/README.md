# Cookbook for running the application
## Tech Stack used:
- Frontend - Streamlit
- LLM Orchestrator - HuggingFace Transformer
- LLM Model - Mistral 7B (v0.3)

### How to run application
The application can be run in the following ways:
- Using the IPYNB notebook which can be run in Google Colab.
- Using the `main.py` file on local machine

#### Google Colab (IPYNB Approach):
- This approach uses Colab's ability to be integrated with a local tunnel to run the Streamlit based UI applcation. 
- The command `!wget -q -O - ipv4.icanhazip.com` gives the IPv4 address of the machine for local tunnel.
- The entire code is stored in app.py file. 
- When `!huggingface-cli login`, it will prompt for the huggingface api token.
- The final command `!streamlit run app.py & npx localtunnel --port 8501` will run the script app.py and activate localtunnel.
- The final URL can be used to access the app and the IPv4 address of the machine should be used as the password to access the URL. 

##### Pre-requisites:
1. HuggingFace API token.
2. DetectionTypes.yml file in Colab environment.

##### Steps to be run the Notebook:
1. The [GenAI_LPMC.ipynb](https://github.com/ewfx/gaied-lpmcai/blob/main/code/src/GenAI_LPMC.ipynb) can be directly opened in Colab using the `Open in Colab` button at the top.
2. HuggingFace API token is needed to access the Mistral model from HuggingFace repository.
3. The API can be obtained from Access Token section of the [HuggingFace Portal](https://huggingface.co/settings/tokens)
4. Add the API to the Colab **Secret** section with the key name of **`HF_TOKEN`**
5. Ensure the API is enabled so that it can be used by the Colab environment
6. Download the `DetectionTypes.yml` which contains the configurable list of Request and Sub-Request types.
7. Upload the yml file in Colab **Files** section.
8. Once all the above steps are complete, each cell in the notebook can be executed one by one.

#### Script based (main.py approach):
- Install a Virtual environment using the command `python -m venv venv`
- Activate the venv using `venv\Scripts\activate` for Windows or `source venv/bin/activate` for Mac OS/Linux based machines.
- Install the necessary dependencies which are listed in the requirements.txt file using `pip install -r requirements.txt`
- Execute `huggingface-cli login` to authenticate with the Huggingface API token
- Finally run the main.py file using `python main.py`

###### Manual Authentication for HuggingFace Portal
- The following code snippet can be used in scenarios where `huggingface-cli login` is not permitted.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login("<ACCESS_TOKEN>")

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
```