import tkinter as tk
import cv2 
from PIL import Image, ImageTk
import numpy as np
from sketch_scripts.pencilSketch import pencil_sketch

class EditorView(tk.Toplevel):
    def __init__(self, parent, path, on_close):
        super().__init__(parent)
        self.on_close = on_close
        self.title(f"Editor - {path}")


        self.bgr = self._load_image(path)
        self.original_bgr = self.bgr.copy()  # Keep a copy of the original image
        self._build()
        # Handle window close event
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _load_image(self, path):
        """Load the image using OpenCV"""
        bgr = cv2.imread(path)

        # Resize if too large
        h,w = bgr.shape[:2]
        if w>800 or w > 600:
            scale = min(800/w, 600/h)
            bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)))

        return bgr

    def _build(self):
        """Build the editor UI"""

        # Image on the left
        self._image_label = tk.Label(self)
        self._image_label.pack(side = tk.LEFT, padx=10, pady=10)

        # Buttons panel on the right
        panel = tk.Frame(self)
        panel.pack(side=tk.RIGHT, fill="y", padx=10, pady=10)

        tk.Button(panel, text="Reset", command=self._reset_image).pack(fill="x", pady=(0,6))
        tk.Button(panel, text="Pencil Sketch", command=self._activate_pencil_sketch).pack(fill="x", pady=(0,6))

        self._params_frame = tk.Frame(panel)
        self._params_frame.pack(fill="x", pady=(0,6))


        self._refresh_image()


    #--- Effect handlers ---


    #--- Pencil Sketch Effect ---
    def _activate_pencil_sketch(self):
        """Activate pencil sketch effect and show parameters"""
        self._clear_params()
        tk.Label(self._params_frame, text="Blur Kernel Size:").pack(anchor="w")
        self._kernel_var = tk.IntVar(value=21)
        slider = tk.Scale(
            self._params_frame,
            from_=1, to=99,
            resolution=2,
            orient="horizontal",
            variable=self._kernel_var,
            command=self._on_pencil_sketch_param_changed,
        )
        slider.pack(fill="x", pady=(0,6))
        self._apply_pencil_sketch()  # Apply effect immediately  

    def _apply_pencil_sketch(self):
        """Apply pencil sketch effect to the image"""
        kernel = self._kernel_var.get()
        self.bgr = pencil_sketch(self.original_bgr, blur_ksize=kernel)  # Use the original image
        self._refresh_image()

    def _on_pencil_sketch_param_changed(self, _event = None):
        """Re-apply pencil sketch effect when parameters change"""
        self._apply_pencil_sketch()



    #--- Utility methods ---

    def _clear_params(self):
        """Clear the parameters frame"""
        for widget in self._params_frame.winfo_children():
            widget.destroy()

    def _refresh_image(self):
        """Convert current BGR image to PIL and display in the label"""
        rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(pil_image)
        self._image_label.config(image=self._photo)
    
    def _reset_image(self):
        """Reset to the original image"""
        self.bgr = self.original_bgr.copy()
        self._clear_params()
        self._refresh_image()

    def close(self):
        """Handle closing the editor"""
        self.destroy()
        if self.on_close:
            self.on_close()
    