#!/usr/bin/env python3
"""
Telegram Bot for Photo Clothing Removal using DreamPower GAN.

This bot downloads user photos, checks for exactly one person in the image,
removes clothing via the open‑source DreamPower model, and sends back the
modified picture. It runs on a GPU (8+ GB VRAM) by default, falling back to
CPU with a warning if CUDA is not available.

HOW TO OBTAIN A BOT TOKEN:
    1. Open Telegram and search for @BotFather.
    2. Send /newbot and follow the instructions.
    3. Copy the token you receive (e.g., 123456:ABC-DEF1234gh...).
    4. Export it: export TELEGRAM_BOT_TOKEN="your_token_here"

HOW TO RUN:
    pip install -r requirements.txt
    export TELEGRAM_BOT_TOKEN="your_token_here"
    python bot.py
"""

import os
import sys
import time
import logging
from io import BytesIO
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
    sys.exit(1)

MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dreampower")
MODEL_FILE = os.path.join(MODEL_CACHE_DIR, "generator.pth")
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DELAY_SECONDS = 5

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

if DEVICE == "cpu":
    logger.warning("CUDA not available. Running on CPU – inference will be slow.")

# ----------------------------------------------------------------------
# DreamPower Generator Architecture (U-Net with ResNet blocks)
# ----------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_res=9):
        super().__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        # Residual blocks
        mult = 2**n_downsampling
        for _ in range(n_res):
            model += [ResidualBlock(ngf * mult)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                    padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True),
            ]

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------------------------
# Model & Person Detector (loaded once at startup)
# ----------------------------------------------------------------------
generator = None
person_detector = None


def download_model():
    """Download the DreamPower generator weights from Hugging Face if not cached."""
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    if not os.path.isfile(MODEL_FILE):
        logger.info("Downloading DreamPower model...")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.error(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )
            sys.exit(1)

        hf_hub_download(
            repo_id="dreamlike-art/dreampower-gan",
            filename="generator.pth",
            local_dir=MODEL_CACHE_DIR,
            local_dir_use_symlinks=False,
        )
        logger.info(f"Model downloaded to {MODEL_FILE}")
    else:
        logger.info("Using cached model.")


def load_model():
    """Load the Generator from the downloaded weights and move to device."""
    global generator
    generator = Generator().to(DEVICE)
    try:
        state = torch.load(MODEL_FILE, map_location=DEVICE)
        generator.load_state_dict(state)
    except Exception as e:
        logger.error(f"Failed to load generator weights: {e}")
        sys.exit(1)
    generator.eval()
    logger.info("Generator loaded successfully.")


def load_person_detector():
    """Load YOLOv5 nano from torch hub for person detection."""
    global person_detector
    logger.info("Loading YOLOv5 nano person detector...")
    person_detector = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
    person_detector.to(DEVICE)
    person_detector.eval()
    logger.info("Person detector ready.")


# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------
def detect_person(image_pil: Image.Image) -> bool:
    """Return True if exactly one person is detected in the image."""
    img_np = np.array(image_pil.convert("RGB"))
    results = person_detector(img_np)
    detections = results.pandas().xyxy[0]
    # class 0 = person in COCO
    persons = detections[detections["class"] == 0]
    num_persons = len(persons)
    logger.info(f"Detected {num_persons} person(s).")
    return num_persons == 1


def remove_clothes(image_pil: Image.Image) -> Image.Image:
    """
    Run the DreamPower inference pipeline:
    1. Resize to 512x512 and convert to tensor
    2. Normalize to [-1, 1]
    3. Run generator
    4. Denormalize to [0, 255], resize back to original dimensions
    """
    original_size = image_pil.size  # (w, h)
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # Denormalize: [-1,1] -> [0,1]
    output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)
    # Convert to PIL
    output_pil = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
    # Resize back to original dimensions
    output_pil = output_pil.resize(original_size, Image.LANCZOS)
    return output_pil


# ----------------------------------------------------------------------
# Telegram handlers
# ----------------------------------------------------------------------
# Simple per‑user rate limiter (5 seconds between requests)
user_last_request = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message."""
    welcome_text = (
        "👋 Welcome to the Clothes Removal Bot!\n\n"
        "Send me a photo containing exactly **one person** (full body works best). "
        "I will process it and return the result.\n\n"
        "⚠️ Please note: processing may take a few seconds. "
        "Do not abuse the service."
    )
    await update.message.reply_text(welcome_text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main photo handler for both compressed and uncompressed images."""
    user_id = update.effective_user.id
    now = time.time()

    # Rate limiting
    if user_id in user_last_request and (now - user_last_request[user_id]) < DELAY_SECONDS:
        remaining = int(DELAY_SECONDS - (now - user_last_request[user_id]))
        await update.message.reply_text(
            f"⏳ Please wait {remaining} second(s) before sending another photo."
        )
        return

    user_last_request[user_id] = now

    try:
        # Download the highest quality version of the photo
        if update.message.photo:
            # Compressed photo (list of sizes, largest is last)
            file = await update.message.photo[-1].get_file()
        elif update.message.document:
            # Uncompressed image document
            if update.message.document.mime_type.startswith("image/"):
                file = await update.message.document.get_file()
            else:
                await update.message.reply_text("Please send an image (JPEG or PNG).")
                return
        else:
            # Should not happen due to filter, but just in case
            return

        # Download image bytes
        image_bytes = await file.download_as_bytearray()
        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")

        await update.message.reply_text("🔍 Processing your photo...")

        # Check for exactly one person
        if not detect_person(image_pil):
            await update.message.reply_text(
                "❌ Please send a photo with exactly one person. "
                "Multiple people or no person detected."
            )
            return

        # Remove clothes
        result_pil = remove_clothes(image_pil)

        # Send result back
        bio = BytesIO()
        result_pil.save(bio, format="JPEG", quality=95)
        bio.seek(0)
        await update.message.reply_photo(
            photo=bio,
            caption="✅ Here is your processed image."
        )
        logger.info(f"Successfully processed photo from user {user_id}")

    except Exception as e:
        logger.exception("Error processing image")
        await update.message.reply_text(
            "⚠️ An error occurred while processing your image. Please try again later."
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # Download model and load it + person detector
    download_model()
    load_model()
    load_person_detector()

    # Build application
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Handlers
    app.add_handler(CommandHandler("start", start))
    # Accept both compressed photos and uncompressed image documents
    app.add_handler(
        MessageHandler(
            filters.PHOTO | filters.Document.IMAGE,
            handle_photo,
        )
    )

    # Start polling
    logger.info("Bot started. Listening for messages...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()


# ======================================================================
# requirements.txt
# ======================================================================
# python-telegram-bot==20.7
# torch>=2.0
# torchvision>=0.15
# opencv-python-headless>=4.8
# Pillow>=10.0
# numpy>=1.24
# httpx>=0.24
# huggingface_hub>=0.16
# pyyaml>=6.0
