# Face Aligner Timelapse Generator

Ever tried making one of those "a photo a day" timelapses, only to realize your head bobs around so much in the final video that it gives you a headache? 

This project fixes that. It takes a folder full of messy, unaligned selfies, uses some clever face-tracking math to lock your eyes in the exact same spot in every picture, and stitches them into a buttery-smooth video.

## The Cool Features

* **Smart Alignment:** It looks for your eyes and nose, then scales and rotates the picture so your face is always perfectly centered.
* **The COVID Mask Fallback:** What if you're wearing a mask and it can't see your nose? No problem. The script is smart enough to realize this and falls back to a 2-point "eyes-only" alignment so your video doesn't break.
* **No Ugly Black Borders:** When you rotate a photo, you get empty black corners. The video generator has a built-in `blur-fill` mode that uses a blurred version of your photo to fill the background (like you see on TikTok or YouTube Shorts). 

---

## Quick Start

### 1. Install Requirements
Make sure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Set Up Your Folders
Create a folder named `raw_photos` in the main directory and dump all your selfies in there. The script will automatically sort them alphabetically/chronologically.

```text
.
├── aligned_photos/   (Script will create this)
├── raw_photos/       <-- PUT YOUR PICS HERE
├── videos/           (Script will create this)
├── requirements.txt
└── scripts/
    ├── runner.py
    └── make_video.py
```

---

## Step 1: Align the Photos

Once your photos are in the `raw_photos` folder, run the alignment engine:

```bash
python scripts/runner.py
```

Sit back and let it run. It will scan every photo, find your face, align it, and save the fixed versions into a new folder called `aligned_photos`. It'll also print out a handy log telling you how many photos it successfully aligned and if it had to use the mask fallback!

---

## Step 2: Make the Video

Now it's time to stitch those aligned photos into a video. Run the video maker:

```bash
python scripts/make_video.py --mode blur-fill
```

### Video Modes Explained

You can change how the final video looks by changing the `--mode` flag:

* **`--mode blur-fill` (Highly Recommended):** Extracts your photo, scales it up, blurs it, and puts the sharp version on top. This completely eliminates any black borders and looks incredibly professional.
* **`--mode strict-crop`:** Crops a tight square around your face. **Warning:** If your face was too close to the edge in the original photo, this mode will throw that photo away to prevent showing any black edges.
* **`--mode vanilla`:** The absolute basics. Just stitches the aligned photos together exactly as they are. You will probably see black borders dancing around the edges.

### Changing the Speed
By default, the video runs at 15 Frames Per Second (FPS). Want it faster? Add the `--fps` flag:

```bash
python scripts/make_video.py --mode blur-fill --fps 24
```

---

## Notes
* **Target Size:** By default, it outputs a vertical 1080x1920 video (perfect for phones). 
* **Expressions:** This tool intentionally ignores your mouth when aligning. This ensures that smiling, frowning, or talking won't cause the image to warp or shake!
