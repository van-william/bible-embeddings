from dotenv import load_dotenv
load_dotenv()

import os
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')
org_id = os.getenv('OPENAI_ORG_ID')
if org_id:
    openai.organization = org_id
    print(f"Using OpenAI organization: {org_id}")
else:
    print("No OPENAI_ORG_ID set; using default organization.")

models = ['text-embedding-3-large', 'text-embedding-ada-002']

for model in models:
    print(f"\nTesting model: {model}")
    try:
        resp = openai.embeddings.create(input="hello world", model=model)
        print(f"✅ Success! Model: {model}")
        print(f"Embedding length: {len(resp.data[0].embedding)}")
        print(f"Usage: {getattr(resp, 'usage', 'N/A')}")
    except Exception as e:
        print(f"❌ Error with model {model}: {e}") 