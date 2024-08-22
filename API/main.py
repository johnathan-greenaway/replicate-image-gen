import os
import asyncio
import logging
import httpx
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import replicate
import sys
import json
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse



# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Initialize FastAPI app
app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use the absolute path for the static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")

# Serve static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))


# Initialize Replicate client
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
REPLICATE_API_URL = "https://api.replicate.com/v1/models"  # Corrected URL definition

# In-memory cache for model versions
MODEL_VERSIONS = {}

# Pydantic models
class ImagePrompt(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    model: str = "davisbrown/flux-half-illustration:687458266007b196a490e79a77bae4b123c1792900e1cb730a51344887ad9832"

# Utility function to serialize model information
def serialize_model(model):
    latest_version = model.get("latest_version", {})
    return {
        "name": model['name'],
        "owner": model['owner'],
        "description": model['description'],
        "visibility": model['visibility'],
        "url": model['url'],
        "version_id": latest_version.get('id', None),
        "cover_image_url": model.get('cover_image_url', ''),
        "github_url": model.get('github_url', None),
        "paper_url": model.get('paper_url', None),
        "license_url": model.get('license_url', None),
        "run_count": model['run_count'],
    }

# Endpoint to list models
@app.get("/list_models/")
async def list_models(page_limit: int = 5):
    try:
        headers = {"Authorization": f"Token {REPLICATE_API_TOKEN}"}
        models = []
        next_url = REPLICATE_API_URL
        page_count = 0

        async with httpx.AsyncClient() as client:
            while next_url and page_count < page_limit:
                response = await client.get(next_url, headers=headers)
                response.raise_for_status()
                response_data = response.json()

                if "results" in response_data:
                    for model in response_data['results']:
                        models.append(model)
                        MODEL_VERSIONS[f"{model['owner']}/{model['name']}"] = model['latest_version']['id']
                else:
                    logger.error(f"Unexpected API response structure: {response_data}")
                    raise HTTPException(status_code=500, detail="Unexpected API response structure.")

                next_url = response_data.get("next")
                page_count += 1

        serialized_models = [serialize_model(model) for model in models]
        return JSONResponse(content={"models": serialized_models})

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

# Function to get the latest version ID for a given model
async def get_latest_version_id(model_name: str) -> str:
    url = f"{REPLICATE_API_URL}/models/{model_name}"
    headers = {"Authorization": f"Token {REPLICATE_API_TOKEN}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch model details")
        
        data = response.json()
        latest_version = data.get('latest_version')
        if not latest_version:
            raise HTTPException(status_code=404, detail="No version found for this model")
        
        return latest_version['id']

# Endpoint to generate an image based on a prompt
@app.post("/generate_image/")
async def generate_image(request: Request):
    try:
        body = await request.json()
        logger.debug(f"Received request body: {body}")

        model_name = body.get("model")
        prompt = body.get("prompt")

        if not model_name or not prompt:
            raise HTTPException(status_code=400, detail="Model name and prompt are required")

        version = MODEL_VERSIONS.get(model_name)
        if not version:
            raise HTTPException(status_code=404, detail=f"Version not found for model '{model_name}'. Please refresh the model list.")

        logger.debug(f"Generating image with model: {model_name}, version: {version}, and prompt: {prompt}")

        # Prepare the request for Replicate API
        replicate_request = {
            "version": version,
            "input": {
                "prompt": prompt
            }
        }

        # Add any additional parameters to the input object
        for key in ["aspect_ratio", "output_format", "guidance_scale", "output_quality"]:
            if key in body:
                replicate_request["input"][key] = body[key]

        headers = {"Authorization": f"Token {REPLICATE_API_TOKEN}"}
        url = "https://api.replicate.com/v1/predictions"

        # Send the request to Replicate API
        async with httpx.AsyncClient() as client:
            logger.debug(f"Sending request to Replicate API: {replicate_request}")
            response = await client.post(url, headers=headers, json=replicate_request)
            logger.info(f"Prediction API status code: {response.status_code}")
            logger.info(f"Prediction API response: {response.text}")

        if response.status_code != 201:
            error_detail = response.json()
            logger.error(f"Error response from Replicate API: {error_detail}")
            return JSONResponse(
                status_code=response.status_code,
                content={"error": error_detail}
            )

        response_data = response.json()
        
        # Poll for the prediction result
        prediction_url = response_data['urls']['get']
        async with httpx.AsyncClient() as client:
            while True:
                poll_response = await client.get(prediction_url, headers=headers)
                poll_data = poll_response.json()
                if poll_data['status'] == 'succeeded':
                    image_url = poll_data['output'][0]
                    break
                elif poll_data['status'] in ['failed', 'canceled']:
                    raise HTTPException(status_code=400, detail=f"Image generation {poll_data['status']}")
                await asyncio.sleep(1)  # Wait for a second before polling again

        logger.debug("Returning response from generate_image function")
        return JSONResponse(content={"image_url": image_url})

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Unexpected error in generate_image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
