
# Persuasive Speech Keyframe Extractor

This project extracts **keyframes** from a speech video based on two conditions:

1. **Voice pitch spikes** → when the speaker raises their voice (indicating emphasis).
2. **Gesture intensity** → when the speaker’s body movements are more extreme (indicating emphasis).

The output is a list of **timestamps** and optionally the **saved frames** from the video.

---

## 📥 Setup

1. Clone or download this repository.

2. Make sure you have **Python 3.11** (recommended, since `mediapipe` does not support Python 3.13 yet).

3. Create and activate a virtual environment:

   ```bash
   python3.11 -m venv venv
   source venv/bin/activate   # Mac/Linux
   .\venv\Scripts\activate    # Windows
   ```

4. Install dependencies using the `requirements.txt` file:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🎥 Download Sample Video

Download this sample test video:

➡️ [input.mp4](https://drive.google.com/file/d/1HLCkd9a6fdO8CBWMq8vORHGziBjxaWcO/view?usp=sharing)

Place it in the **project root folder** and rename it to:

```
input.mp4
```

---

## 🚀 Usage

Run the script with:

```bash
python main.py input.mp4 --save-frames
```

* `input.mp4` → path to your video file.
* `--save-frames` → optional flag to save extracted keyframes as `.jpg` images.

---

## 📊 Output

* Console will print a list of **timestamps (in seconds)** for detected keyframes.
* If `--save-frames` is passed, frames will be saved in the current directory as:

```
frame_<timestamp>.jpg
```

Example:

```
=== Keyframe Timestamps (sec) ===
[3.1, 7.5, 12.8, 20.2]
```

---

## 📄 requirements.txt

```txt
opencv-python
numpy
librosa
mediapipe
scipy
```

---

## 🛠 Notes

* Thresholds for voice pitch and gesture intensity can be tuned in `main.py`.
* Works best on videos with **clear audio and visible speaker**.

