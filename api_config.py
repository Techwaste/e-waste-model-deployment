from fastapi import FastAPI

description = """
Endpoints for anything related to Image Classifications

You will be able to:

* **Preprocessing Image** (implemented).
* **Predict Image** (implemented).
"""

tags_metadata = [
    {
        "name": "preprocessing",
        "description": "Preprocessing an Image to match the model",
    },
    {
        "name": "predict",
        "description": "Predict an Image",
    },
]

app = FastAPI(
    title="TechWas-PredictionAPI",
    description=description,
    version="0.0.1",
    # terms_of_service="http://example.com/terms/",
    contact={
        "name": "TechWas Team",
        "url": "https://github.com/TechWas",
        # "email": "dp@x-force.example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/license/mit/",
    },
    openapi_tags=tags_metadata,
)
