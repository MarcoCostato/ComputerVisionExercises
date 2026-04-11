import tkinter as tk
import cv2 
from PIL import Image, ImageTk
import numpy as np

class EditorView(tk.Toplevel):
    def __init__(self, parent, path, on_close):
        super().__init__(parent)
        self.on_close = on_close
        self.title(f"Editor - {path}")

        self._load_image(path)
        self._build()
        # Handle window close event
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _load_image(self, path):
        """Load the image using OpenCV and convert to PIL format"""
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        h,w = rgb.shape[:2]
        if w>800 or w > 600:
            scale = min(800/w, 600/h)
            rgb = cv2.resize(rgb, (int(w*scale), int(h*scale)))

        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))

    def _build(self):
        """Build the editor UI"""
        tk.Label(self, image=self._photo).pack(padx=10, pady=10)

    def close(self):
        """Handle closing the editor"""
        self.destroy()
        if self.on_close:
            self.on_close()
    