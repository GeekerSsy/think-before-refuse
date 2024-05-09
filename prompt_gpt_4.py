import requests
from tqdm.asyncio import tqdm
import logging
import os
import concurrent.futures
import asyncio
import aiohttp
import openai

import argparse
import re
import json
import csv
import numpy as np
openai.api_base = "https://gda-oai-uks.openai.azure.com"
# openai.api_base = "https://gda-oai.openai.azure.com/"

openai.api_type = "azure"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = "2023-09-15-preview"

response = openai.ChatCompletion.create(
    engine="gda-aoi-gpt4-32k",  # replace with your engine
    messages=[
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user},
    ]
)

response_score = response['choices'][0]['message']['content']

print(response_score)
