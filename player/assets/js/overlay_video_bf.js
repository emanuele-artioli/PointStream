const backgroundCanvas = document.getElementById("backgroundCanvas");
const backgroundCtx = backgroundCanvas.getContext("2d");
const foregroundCanvas = document.getElementById("foregroundCanvas");
const foregroundCtx = foregroundCanvas.getContext("2d");

// Frames per second (FPS)
const FPS = 4;

// Define the images and their properties
const backgroundImageSrc = "assets/imgs/background.png";
const foregroundTemplates = ["assets/imgs/player1/", "assets/imgs/player2/"];
const bufferSize = 8; // Maintain at least 8 frames in the buffer
const imageExtension = ".png";

// Preloaded images buffer
let backgroundImage;
const preloadedForegroundBuffers = [new Map(), new Map()];
const frameCounters = [0, 0]; // Track frame numbers for each template

// Preloaded foreground images properties (updated dynamically)
const foregroundImages = [
    { x: 160, y: 20, frameIndex: 0, templateIndex: 0, playerId: 1, img: null },
    { x: 50, y: 190, frameIndex: 0, templateIndex: 1, playerId: 2, img: null },
];

// Preload an image
function preloadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.src = src;
        img.onload = () => resolve(img);
        img.onerror = (err) => reject(err);
    });
}

// Generate a zero-padded frame number string
function getFrameNumber(index) {
    return String(index).padStart(4, "0");
}

// Fetch a segment image and its coordinates from the server based on player ID and frame index
async function fetchSegmentImage(playerId, frameIndex) {
    try {
        const response = await fetch(`http://localhost:80/frame/player${playerId}/${frameIndex}`);
        if (!response.ok) {
            throw new Error("Failed to fetch segment image");
        }
        const blob = await response.blob();
        const img = await createImageBitmap(blob);
        // Get the x and y coordinates from the response headers
        const x = parseInt(response.headers.get("X-Coordinate")) || 0;  // Fallback coordinates if headers are missing
        const y = parseInt(response.headers.get("Y-Coordinate")) || 0;  // Fallback coordinates if headers are missing
        // console.log(`x: ${x}, y: ${y}`);

        // Find the corresponding entry in foregroundImages and update it
        const imageEntry = foregroundImages.find((image) => image.playerId === playerId);
        if (imageEntry) {
            imageEntry.img = img;
            imageEntry.x = x;
            imageEntry.y = y;
            imageEntry.width = img.width;   // Set dynamic width
            imageEntry.height = img.height; // Set dynamic height
        }

    } catch (error) {
        console.error("Error fetching segment image:", error);
    }
}

// Preload frames in batches (includes fetching segment images)
async function preloadContinuous() {
    try {
        while (true) {
            // Preload the foreground images and fetch segment images concurrently
            await Promise.all(foregroundTemplates.map(async (template, t) => {
                const frameIndex = frameCounters[t] % bufferSize;
                const frameSrc = `${template}${getFrameNumber(frameCounters[t])}${imageExtension}`;
                const img = await preloadImage(frameSrc);
                
                // Save image reference to the foregroundImages array and update dimensions dynamically
                if (!foregroundImages[t].img) {
                    foregroundImages[t].img = img;
                    foregroundImages[t].width = img.width;   // Set dynamic width
                    foregroundImages[t].height = img.height; // Set dynamic height
                }
                
                preloadedForegroundBuffers[t].set(frameIndex, img);
                if (preloadedForegroundBuffers[t].size > bufferSize) {
                    preloadedForegroundBuffers[t].delete(frameIndex - bufferSize);
                }

                // Fetch segment image for the corresponding player and frame index
                await fetchSegmentImage(t + 1, frameCounters[t]);

                frameCounters[t] = (frameCounters[t] + 1) % bufferSize;
            }));

            await new Promise(resolve => setTimeout(resolve, 1000 / FPS)); // Adjust fetch interval
        }
    } catch (error) {
        console.error("Error preloading images:", error);
    }
}

// Draw the background on the background canvas
function drawBackground() {
    backgroundCtx.drawImage(backgroundImage, 0, 0, backgroundCanvas.width, backgroundCanvas.height);
}

// Draw the foreground images on the foreground canvas
function drawForeground() {
    foregroundCtx.clearRect(0, 0, foregroundCanvas.width, foregroundCanvas.height);
    foregroundImages.forEach((image) => {
        if (image.img) {
            // Use dynamic width and height
            foregroundCtx.drawImage(image.img, image.x, image.y, image.width, image.height);
        }
        // Cycle through frames in buffer
        image.frameIndex = (image.frameIndex + 1) % bufferSize;
    });
}

// Main animation loop using requestAnimationFrame
function animate() {
    drawForeground();
    setTimeout(animate, 1000 / FPS);
}

// Start the animation after preloading all images
preloadImage(backgroundImageSrc).then(img => {
    backgroundImage = img;
    drawBackground();
    preloadContinuous();
    animate();
});
