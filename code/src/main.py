import streamlit as st
import email
import hashlib
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
import jaydebeapi
import pdfplumber
import pypandoc
from docx import Document
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup H2 DB and store it in Streamlit session to ensure DB connections are not loaded up in every render of application
def setup_h2_database():
    if "h2_connection" not in st.session_state:
        logger.info("Setting up H2 database...")
        try:
            conn = jaydebeapi.connect(
                "org.h2.Driver",
                "jdbc:h2:~/email_db",
                ["sa", ""],
                "h2.jar"
            )
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emails (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    filename VARCHAR(255),
                    sender VARCHAR(255),
                    request_type VARCHAR(255),
                    sub_request_type TEXT,
                    confidence VARCHAR(255),
                    email_hash VARCHAR(255)
                )
            """)

            # Store in session state to prevent reloading
            st.session_state.h2_connection = conn
            st.session_state.h2_cursor = cursor
            logger.info("H2 database setup complete.")
        except Exception as e:
            logger.error(f"Error setting up H2 database: {str(e)}")
            st.session_state.h2_connection = None
            st.session_state.h2_cursor = None

    return st.session_state.h2_connection, st.session_state.h2_cursor

# Load and cache the model using Streamlit cache resource annotation to ensure app doesn't reload during every render
@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit quantization
        bnb_4bit_compute_dtype=torch.float16,  # Use float16 for computation
        bnb_4bit_quant_type="nf4",  # More efficient quantization type
        bnb_4bit_use_double_quant=True  # Extra compression
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
    "requestType": "string",
    "subRequestTypes": ["string"],
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
    attention_mask=attention_mask,  # Pass attention_mask explicitly
    max_length=4096,
    pad_token_id=tokenizer.pad_token_id  # Use correct pad_token_id
  )

  # Decode and Extract JSON Output
  response = tokenizer.decode(output[0], skip_special_tokens=True)

  # Extract only the JSON output
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
def classify_and_extract(input_text):
    classifiedResult = classify(input_text)
    extractedResult = extract(input_text)
    logger.info(f"Final classification results: {classifiedResult}")
    logger.info(f"Final extraction results: {extractedResult}")

    classifiedResult["nameValuePairs"] = extractedResult
    print(classifiedResult)
    return classifiedResult

# Directory to store attachments
ATTACHMENTS_DIR = "attachments/"
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)

# Helper functions to parse the attachments based on their file type
def process_attachment(file_path):
    """Reads and extracts text from PDF, DOCX, DOC, and TXT files."""
    try:
        if file_path.endswith(".pdf"):
            return extract_text_from_pdf(file_path)
        elif file_path.endswith(".docx"):
            return extract_text_from_docx(file_path)
        elif file_path.endswith(".doc"):
            return extract_text_from_doc(file_path)
        elif file_path.endswith(".txt"):
            return extract_text_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error processing attachment {file_path}: {e}")
        return None


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text if text else "(No readable text in PDF)"
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        return None


def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text if text else "(No readable text in DOCX)"
    except Exception as e:
        logger.error(f"Error reading DOCX {docx_path}: {e}")
        return None


def extract_text_from_doc(doc_path):
    """Extracts text from a DOC file using pypandoc."""
    try:
        return pypandoc.convert_file(doc_path, "plain")
    except Exception as e:
        logger.error(f"Error reading DOC {doc_path}: {e}")
        return None


def extract_text_from_txt(txt_path):
    """Extracts text from a TXT file."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading TXT {txt_path}: {e}")
        return None
    
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
            content_type = part.get_content_type()

            if content_type == "text/plain":
                body += part.get_payload(decode=True).decode(errors="ignore")
            elif content_type == "text/html":
                html_content = part.get_payload(decode=True).decode(errors="ignore")
                soup = BeautifulSoup(html_content, "html.parser")
                body += soup.get_text(separator=" ") + "\n"
            elif part.get_content_disposition() == "attachment":
                attachment_name = part.get_filename()
                attachment_data = part.get_payload(decode=True)

                if not attachment_name or not attachment_data:
                    continue  # Skip if no valid attachment

                # Save the attachment file locally
                file_path = os.path.join(ATTACHMENTS_DIR, attachment_name)
                with open(file_path, "wb") as f:
                    f.write(attachment_data)

                # Try to extract text from known file types
                extracted_text = process_attachment(file_path)

                if extracted_text:
                    attachment_type = "text"
                    attachment_content = extracted_text  # Store extracted text
                else:
                    attachment_type = "binary"
                    attachment_content = base64.b64encode(attachment_data).decode()  # Store Base64

                logger.info(f"Attachment found: {attachment_name} (Type: {attachment_type}, Size: {len(attachment_data)} bytes)")

                attachments.append({
                    "name": attachment_name,
                    "type": attachment_type,
                    "content": attachment_content,
                    "size": len(attachment_data)
                })

        email_data = {
            "subject": subject,
            "from": sender,
            "to": recipient,
            "date": date,
            "body": body,
            "attachments": attachments  # List of extracted attachments
        }

        logger.info(f"Email components extracted: {subject}")
        return email_data

    except Exception as e:
        logger.error(f"Error parsing email: {e}")
        return {"error": str(e)}

# Computes the hash using the email sender and email content
def compute_hash(email_data):
    text = email_data['subject'] + email_data['body']
    email_hash = hashlib.md5(text.encode()).hexdigest()
    logger.info(f"Hash computed: {email_hash}")

# Detects duplicates from DB using email sender and email hash
def detect_duplicates(cursor, email_data, email_hash):
    logger.info(f"Detecting duplicates for email: {email_data['filename']}")
    try:
        cursor.execute("SELECT COUNT(*) FROM emails WHERE sender = ? and email_hash = ?", (email_data['from'], email_hash,))
        count = cursor.fetchone()[0]
        is_duplicate = count > 0
        logger.info(f"Duplicate check: {'Duplicate' if is_duplicate else 'Not a duplicate'}")
        return is_duplicate
    except Exception as e:
        logger.error(f"Error detecting duplicates for {email_data['filename']}: {str(e)}")
        return False
    
# Store the results for an email in H2 DB
def store_email_data(cursor, email_data, output, email_hash):
    logger.info(f"Storing email data for: {email_data['filename']}")
    try:
        cursor.execute("""
            INSERT INTO emails (filename, sender, request_type, sub_request_type, confidence, email_hash) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            email_data['filename'],
            email_data['from'],
            output['result']['output']['requestType'],
            json.dumps(output['result']['output']['subRequestTypes']),
            output['result']['probability'],
            email_hash
        ))
        logger.info("Email data stored successfully.")
    except Exception as e:
        logger.error(f"Error storing email data for {email_data['filename']}: {str(e)}")

# Streamlit app title
st.title("Email content classifier")
uploaded_file = st.file_uploader("Upload an .eml file", type=["eml"])

if uploaded_file:
    with uploaded_file:
        filename = uploaded_file.name
        uploaded_file.seek(0)  # Reset file pointer before reading
        email_data, attachments = parse_eml(uploaded_file)
        email_data["filename"] = filename
        email_hash = compute_hash(email_data)
        conn, cursor = setup_h2_database()
        if conn and cursor:
            isDuplicate = detect_duplicates(cursor, email_data, email_hash)
            if isDuplicate:
                logger.error(f"Dupicate detected. Mail contents already exists in DB")
            else:
                with st.spinner(f"Processing email {filename}... Please wait ⏳"):
                    result = classify_and_extract(email_data["body"])
                    store_email_data(cursor, email_data, result, email_hash)
                    conn.comit()
                    output = {
                        "emailContents": email_data,
                        "result": result
                    }
                    st.json(output)