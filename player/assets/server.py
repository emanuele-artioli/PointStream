from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from typing import Optional, List

# Define the FastAPI application
app = FastAPI()

# Set CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, adjust for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulate a directory containing images
IMAGE_DIR = "imgs"  # Adjust path as needed
IMAGE_EXTENSION = ".png"

def get_coordinates(player_id, frame_index):
    x = 0
    y = 0
    if player_id == 1:
        x = 160
        y = 20
    elif player_id == 2:
        x = 50
        y = 190
    return x, y

# FastAPI GET request handler for segment images
@app.get("/frame/player{player_id}/{frame_index}")
async def get_segment_image(player_id: int, frame_index: int):
    # Calculate the x and y coordinates for the segment index
    x, y = get_coordinates(player_id, frame_index)
    # print(f"x: {x}, y: {y}")
    # Generate image filename based on segment index
    image_filename = f"{frame_index:04d}{IMAGE_EXTENSION}"
    image_path = os.path.join(IMAGE_DIR, f"player{player_id}", image_filename)
    # print(f"Image Path: {image_path}")

    # Check if image exists
    if os.path.exists(image_path) and os.path.isfile(image_path):
        # Return the image with x, y coordinates
        response = FileResponse(image_path)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "X-Coordinate, Y-Coordinate"
        response.headers["Access-Control-Expose-Headers"] = "X-Coordinate, Y-Coordinate"
        # Add x and y as headers
        response.headers["X-Coordinate"] = str(x)
        response.headers["Y-Coordinate"] = str(y)
        return response
    else:
        raise HTTPException(status_code=404, detail=f"Image for frame {frame_index} not found.")
        
# FastAPI GET request handler
@app.get("/{path:path}")
async def handle_get(path: str, request: Request):
    file_path = path # request.url.path

    if file_path == "/":
        file_path = index_path
        
    # Simulating file reading (or replace with your actual file handling logic)
    content = None
    if os.path.exists(file_path) and os.path.isfile(file_path):
        
        response = FileResponse(file_path)
            
        response.headers["Access-Control-Allow-Origin"] = "*"
        
        return response
    else:
        raise HTTPException(status_code=404, detail=f"File {file_path} not found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)