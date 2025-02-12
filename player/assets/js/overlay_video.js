const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Frames per second (FPS)
const FPS = 30;

// Define the images and their properties
const backgroundImageSrc = "assets/imgs/background.png";
const foregroundTemplates = ["assets/imgs/player1/", "assets/imgs/player2/"];
// const bufferSize = 30; // Maintain at least 30 frames in the buffer
const bufferSize = 8; // Maintain at least 30 frames in the buffer
const imageExtension = ".png";

const foregroundImages = [
    { x: 100, y: 150, width: 100, height: 100, dx: 2, dy: 1, frameIndex: 0, templateIndex: 0 },
    { x: 300, y: 200, width: 120, height: 120, dx: -1, dy: 2, frameIndex: 0, templateIndex: 1 },
];

// Preloaded images buffer
let backgroundImage;
const preloadedForegroundBuffers = [new Map(), new Map()];
const frameCounters = [0, 0]; // Track frame numbers for each template

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

// Continuously fetch new frames
async function preloadContinuous() {
    try {
        while (true) {
            for (let t = 0; t < foregroundTemplates.length; t++) {
                const frameIndex = frameCounters[t] % bufferSize;
                const frameSrc = `${foregroundTemplates[t]}${getFrameNumber(frameCounters[t])}${imageExtension}`;
                const img = await preloadImage(frameSrc);
                preloadedForegroundBuffers[t].set(frameIndex, img);
                if (preloadedForegroundBuffers[t].size > bufferSize) {
                    preloadedForegroundBuffers[t].delete(frameIndex - bufferSize);
                }
                frameCounters[t]++; // Increment frame counter for next fetch
                frameCounters[t] = frameCounters[t] % bufferSize;
            }
            await new Promise(resolve => setTimeout(resolve, 1000 / FPS)); // Adjust fetch interval
        }
    } catch (error) {
        console.error("Error preloading images:", error);
    }
}

// Update positions of foreground images and cycle frames
function updatePositions() {
    for (let image of foregroundImages) {
        image.x += image.dx;
        image.y += image.dy;

        // Check canvas boundaries and reverse direction if needed
        if (image.x < 0 || image.x + image.width > canvas.width) {
            image.dx *= -1;
        }
        if (image.y < 0 || image.y + image.height > canvas.height) {
            image.dy *= -1;
        }

        // Cycle through frames in buffer
        image.frameIndex = (image.frameIndex + 1) % bufferSize;
    }
}

// Draw the images on the canvas
function drawFrame() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);

    foregroundImages.forEach((image) => {
        const fg = preloadedForegroundBuffers[image.templateIndex].get(image.frameIndex);
        if (fg) {
            ctx.drawImage(fg, image.x, image.y, image.width, image.height);
        }
    });
}

// Main animation loop
function animate() {
    drawFrame();
    updatePositions();
    setTimeout(animate, 1000 / FPS);
}

// Start the animation after preloading all images
preloadImage(backgroundImageSrc).then(img => {
    backgroundImage = img;
    preloadContinuous();
    animate();
});