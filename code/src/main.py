import torch
import streamlit as st
import email
import os
import logging
from email import policy
from email.parser import BytesParser
from email.message import Message
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import yaml
import ast

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and cache the model using Streamlit cache resource annotation to ensure app doesn't reload during every render
@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit quantization for model optimization
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True
    )

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    logger.info(f"Loading model from cache...")
    return model, tokenizer

model, tokenizer = load_model()

# Parse the configuration (request, sub-request and keyword) that will be passed as prompt to LLM
def config_parser():
  logger.info(f"Parsing configuration from yml file...")
  with open("DetectionTypes.yml", "r") as f:
    config = yaml.safe_load(f)
  return config

config = config_parser()
request_types = [ request for request in config.keys()]

# Prompt Template for the classification task
classification_prompt = """
You are a expert helping users to categorize the text based on predefined classes. Only categorize the content based on the available list of classes.
Given an input text, classify it into one of the **main request types**: {request_types} and, if applicable, further classify it into a **sub-request** type.
Your response **must only** be a JSON dictionary with: {outputFormat}

### **Classification Rules**:
- **Request Type:** Match the text to the most relevant **Request type** based on its semantics, meaning and the sample list of keywords.
- **Sub-Request Type:** If the text aligns with a sub-category, classify it accordingly.
- The following are some of the example Request-Types, their sample reference keywords and Sub-Request Types.
{config}

Text: "{input_text}"

Output:
"""
# Prompt Template for the Name-Entity Recognition task to identify the key-value pairs
ner_prompt = """
You are a financial expert who is responsible identifying all the important key value pairs associated with Finance/Banking, etc.
Extract financial details from the given content and return them **strictly as a dictionary of key-value pairs**.
Text: "{input_text}"

Output:
"""

# Structure of Classification output
classificationOutputFormat = {
    "output": {
    "category": "string",
    "subcategory": ["string"],
    "probability": "float",
    "reasoning": "string"
  }
}

# Tokenizes the input text (prompt and the email content) into tensors and invokes the model
# Decodes the tensors back to string and returns the response
def invoke(prompt):
  # Ensure pad_token is properly set
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Assign pad_token explicitly

  # Tokenize input with attention mask
  inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
  input_ids = inputs["input_ids"].to(model.device)
  attention_mask = inputs["attention_mask"].to(model.device)

  # Generate output with explicit attention mask
  output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=4096,
    pad_token_id=tokenizer.pad_token_id
  )

  # Decode and Extract JSON Output
  response = tokenizer.decode(output[0], skip_special_tokens=True)
  if "Output:" in response:
      response = response.split("Output:")[-1].strip()

  return response

# Classify takes content of mail as parameter, creates the classification prompt and then calls invoke function to get LLM response
def classify(input_text):
    formatted_prompt = classification_prompt.format(
        input_text=input_text,
        outputFormat=classificationOutputFormat,
        request_types=request_types,
        config=config)
    classifiedOutput = invoke(formatted_prompt)
    logger.info(f"Classified results: {classifiedOutput}")
    return ast.literal_eval(classifiedOutput)

# Extract takes content of mail as parameter, creates the NER prompt and then calls invoke function to get LLM response
def extract(input_text):
    formatted_prompt = ner_prompt.format(input_text=input_text)
    extractedOutput = invoke(formatted_prompt)
    logger.info(f"Extracted results: {extractedOutput}")
    return json.loads(extractedOutput)

# Wrapper for Classify and Extract methods, responsible for concatenating all the output
def classifyAndExtract(input_text):
    # Classification
    classifiedResult = classify(input_text)
    extractedResult = extract(input_text)
    logger.info(f"Final classification results: {classifiedResult}")
    logger.info(f"Final extraction results: {extractedResult}")

    result = {
        "classification": classifiedResult,
        "extraction": extractedResult
    }
    print(result)
    return result

# Parses the file uploaded through UI.
# Walks through all the email content and its attachment and returns them
def parse_eml(file):
    try:
        msg: Message = BytesParser(policy=policy.default).parse(file)
        logger.info("Email content classifier")

        subject = msg["subject"] or "No Subject"
        sender = msg["from"]
        recipient = msg["to"]
        date = msg["date"]
        body = ""
        attachments = []

        # Extract body and attachments
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_payload(decode=True).decode(errors="ignore")
            elif not part.get_content_type().startswith("multipart"):
                attachment_name = part.get_filename()
                attachment_data = part.get_payload(decode=True)
                if not attachment_name or not attachment_data:
                  continue  # Skip if no valid attachment

                attachment_content = None
                attachment_type = "unknown"

                try:
                    # Try decoding as a text file
                    attachment_content = attachment_data.decode(errors="ignore")
                    attachment_type = "text"
                except UnicodeDecodeError:
                    # If not a text file, encode in Base64
                    attachment_content = base64.b64encode(attachment_data).decode()
                    attachment_type = "binary"

                # Log after ensuring `attachment_content` is assigned
                logger.info(f"Attachment found: {attachment_name} (Type: {attachment_type}, Size: {len(attachment_data)} bytes)")

                attachments.append({
                    "name": attachment_name,
                    "type": attachment_type,
                    "content": attachment_content,
                    "size": len(attachment_data),
                    "raw_data": attachment_data
                })

        email_data = {
            "subject": subject,
            "from": sender,
            "to": recipient,
            "date": date,
            "body": body,
            "attachments": [name for name in attachments]
        }
        logger.info(f"Email components extracted: {subject}")

        return email_data, attachments
    except Exception as e:
        logger.error(f"Error parsing email: {e}")
        return {"error": str(e)}, []

# Streamlit UI
st.title("Email content classifier")

uploaded_file = st.file_uploader("Upload an .eml file", type=["eml"])

if uploaded_file:
    with uploaded_file:
        email_data, attachments = parse_eml(uploaded_file)
        with st.spinner("Processing email... Please wait ⏳"):
            result = classifyAndExtract(email_data["body"])
            output = {
                "emailContents": email_data,
                "result": result
            }
            st.json(output)