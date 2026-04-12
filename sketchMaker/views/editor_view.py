import tkinter as tk
from tkinter import filedialog
import cv2 
from PIL import Image, ImageTk
import numpy as np
from sketch_scripts.pencilSketch import pencil_sketch, color_pencil_sketch
from sketch_scripts.basic_processing import sobel_edge_detection, canny_edge_detection, dynamic_gamma_correction
from sketch_scripts.sepia import sepia_filter

class EditorView(tk.Toplevel):
    def __init__(self, parent, path, on_close):
        super().__init__(parent)
        self.on_close = on_close
        self.title(f"Editor - {path}")

        self._debaunce_job = None # To store the after job for debouncing parameter changes

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
        tk.Button(panel, text="Brightness/Contrast", command=self._activate_brightness_contrast).pack(fill="x", pady=(0,6))
        tk.Button(panel, text="Gamma Correction", command=self._activate_gamma_correction).pack(fill="x", pady=(0,6))
        tk.Button(panel, text="Dynamic Gamma Correction", command=self._activate_dynamic_gamma_correction).pack(fill="x", pady=(0,6))
        tk.Button(panel, text="Gaussian Blur", command=self._activate_gaussian_blur).pack(fill="x", pady=(0,6))
        tk.Button(panel, text="Bilateral Filter", command=self._activate_bilateral_filter).pack(fill="x", pady=(0,6))
        tk.Button(panel, text="Sobel Edge Detection", command=self._activate_sobel).pack(fill="x", pady=(0,6))
        tk.Button(panel, text="Canny Edge Detection", command=self._activate_canny).pack(fill="x", pady=(0,6))
        tk.Button(panel, text="Black & White", command=self._activate_black_and_white).pack(fill="x", pady=(0,6))
        tk.Button(panel, text="Sepia", command=self._activate_sepia).pack(fill="x", pady=(0,6))
        tk.Button(panel, text="Pencil Sketch", command=self._activate_pencil_sketch).pack(fill="x", pady=(0,6))
        tk.Button(panel, text="Color Pencil Sketch", command=self._activate_color_pencil_sketch).pack(fill="x", pady=(0,6))
        tk.Button(panel, text="Compose", command=self._compose_image).pack(fill="x", pady=(0,6))
        tk.Button(panel, text="Save as PNG", command=self._save_png).pack(fill="x", pady=(0,6))

        self._params_frame = tk.Frame(panel)
        self._params_frame.pack(fill="x", pady=(0,6))


        self._refresh_image()


    #--- Effect handlers ---


    #--- Brightness/Contrast Effect ---
    def _activate_brightness_contrast(self):
        """Activate brightness/contrast effect and show parameters"""
        self._clear_params()
        tk.Label(self._params_frame, text="Brightness:").pack(anchor="w")
        self._brightness_var = tk.IntVar(value=0)
        brightness_slider = tk.Scale(
            self._params_frame,
            from_=-100, to=100,
            orient="horizontal",
            variable=self._brightness_var,
            command=self._on_brightness_contrast_param_changed,
        )
        brightness_slider.pack(fill="x", pady=(0,6))

        tk.Label(self._params_frame, text="Contrast:").pack(anchor="w")
        self._contrast_var = tk.IntVar(value=0)
        contrast_slider = tk.Scale(
            self._params_frame,
            from_=-100, to=100,
            orient="horizontal",
            variable=self._contrast_var,
            command=self._on_brightness_contrast_param_changed,
        )
        contrast_slider.pack(fill="x", pady=(0,6))

        self._apply_brightness_contrast()  # Apply effect immediately

    def _apply_brightness_contrast(self):
        """Apply brightness/contrast effect to the image"""
        brightness = self._brightness_var.get()
        contrast = self._contrast_var.get()

        # Apply contrast
        factor = (259 * (contrast + 255)) / (255 * (259 - contrast)) if contrast != 0 else 1
        adjusted = cv2.convertScaleAbs(self.original_bgr, alpha=factor, beta=brightness)

        self.bgr = adjusted
        self._refresh_image()

    def _on_brightness_contrast_param_changed(self, _event = None):
        """Re-apply brightness/contrast effect when parameters change"""
        self._apply_brightness_contrast()


    #--- Gamma Correction Effect ---
    def _activate_gamma_correction(self):
        """Activate gamma correction effect and show parameters"""
        self._clear_params()
        tk.Label(self._params_frame, text="Gamma:").pack(anchor="w")
        self._gamma_var = tk.DoubleVar(value=1.0)
        gamma_slider = tk.Scale(
            self._params_frame,
            from_=0.1, to=5.0,
            resolution=0.1,
            orient="horizontal",
            variable=self._gamma_var,
            command=self._on_gamma_correction_param_changed,
        )
        gamma_slider.pack(fill="x", pady=(0,6))
        self._apply_gamma_correction()  # Apply effect immediately

    def _apply_gamma_correction(self):
        """Apply gamma correction effect to the image"""
        gamma = self._gamma_var.get()
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        self.bgr = cv2.LUT(self.original_bgr, table)
        self._refresh_image()

    def _on_gamma_correction_param_changed(self, _event = None):
        """Re-apply gamma correction effect when parameters change"""
        self._apply_gamma_correction()

    #--- Dynamic Gamma Correction Effect ---
    def _activate_dynamic_gamma_correction(self):
        """Activate dynamic gamma correction effect"""
        self._clear_params()
        tk.Label(self._params_frame, text="Blur Kernel Size:").pack(anchor="w")
        self._blur_ksize_var = tk.IntVar(value=21)
        blur_ksize_slider = tk.Scale(
            self._params_frame,
            from_=1, to=99,
            resolution=2,
            orient="horizontal",
            variable=self._blur_ksize_var,
            command=self._on_dynamic_gamma_correction_param_changed,
        )
        blur_ksize_slider.pack(fill="x", pady=(0,6))

        self._apply_dynamic_gamma_correction()  # Apply effect immediately

    def _apply_dynamic_gamma_correction(self):
        """Apply dynamic gamma correction effect to the image"""
        self.bgr = dynamic_gamma_correction(self.original_bgr, blur_ksize=self._blur_ksize_var.get())
        self._refresh_image()

    def _on_dynamic_gamma_correction_param_changed(self, _event = None):
        """Re-apply dynamic gamma correction effect when parameters change WITH DEBOUNCE"""
        if self._debaunce_job:
            self._params_frame.after_cancel(self._debaunce_job)
        self._debaunce_job = self._params_frame.after(50, self._apply_dynamic_gamma_correction)



    

    #--- Gaussian Blur Effect ---
    def _activate_gaussian_blur(self):
        """Activate Gaussian blur effect and show parameters"""
        self._clear_params()
        tk.Label(self._params_frame, text="Kernel Size:").pack(anchor="w")
        self._kernel_var = tk.IntVar(value=5)
        self._sigma_var = tk.IntVar(value=0)
        kSizeSlider = tk.Scale(
            self._params_frame,
            from_=1, to=99,
            resolution=2,
            orient="horizontal",
            variable=self._kernel_var,
            command=self._on_gaussian_blur_param_changed,
        )
        kSizeSlider.pack(fill="x", pady=(0,6))
        tk.Label(self._params_frame, text="Sigma:").pack(anchor="w")
        sigmaSlider = tk.Scale(
            self._params_frame,
            from_=0, to=100,
            orient="horizontal",
            variable=self._sigma_var,
            command=self._on_gaussian_blur_param_changed,
        )
        sigmaSlider.pack(fill="x", pady=(0,6))
        self._apply_gaussian_blur()  # Apply effect immediately
    
    def _apply_gaussian_blur(self):
        """Apply Gaussian blur effect to the image"""
        kernel = self._kernel_var.get()
        sigma = self._sigma_var.get()
        self.bgr = cv2.GaussianBlur(self.original_bgr, (kernel, kernel), sigma)  # Use the original image
        self._refresh_image()

    def _on_gaussian_blur_param_changed(self, _event = None):
        """Re-apply Gaussian blur effect when parameters change"""
        self._apply_gaussian_blur()

    #--- Bilateral Filter Effect ---
    def _activate_bilateral_filter(self):
        """Activate bilateral filter effect and show parameters"""
        self._clear_params()
        tk.Label(self._params_frame, text="Diameter:").pack(anchor="w")
        self._diameter_var = tk.IntVar(value=9)
        diameter_slider = tk.Scale(
            self._params_frame,
            from_=1, to=99,
            resolution=2,
            orient="horizontal",
            variable=self._diameter_var,
            command=self._on_bilateral_filter_param_changed,
        )
        diameter_slider.pack(fill="x", pady=(0,6))

        tk.Label(self._params_frame, text="Sigma Color:").pack(anchor="w")
        self._sigma_color_var = tk.IntVar(value=75)
        sigma_color_slider = tk.Scale(
            self._params_frame,
            from_=1, to=255,
            orient="horizontal",
            variable=self._sigma_color_var,
            command=self._on_bilateral_filter_param_changed,
        )
        sigma_color_slider.pack(fill="x", pady=(0,6))

        tk.Label(self._params_frame, text="Sigma Space:").pack(anchor="w")
        self._sigma_space_var = tk.IntVar(value=75)
        sigma_space_slider = tk.Scale(
            self._params_frame,
            from_=1, to=255,
            orient="horizontal",
            variable=self._sigma_space_var,
            command=self._on_bilateral_filter_param_changed,
        )
        sigma_space_slider.pack(fill="x", pady=(0,6))

        self._apply_bilateral_filter()  

    def _apply_bilateral_filter(self):
        """Apply bilateral filter effect to the image"""
        diameter = self._diameter_var.get()
        sigma_color = self._sigma_color_var.get()
        sigma_space = self._sigma_space_var.get()
        self.bgr = cv2.bilateralFilter(self.original_bgr, diameter, sigma_color, sigma_space)
        self._refresh_image()

    def _on_bilateral_filter_param_changed(self, _event = None):
        """Re-apply bilateral filter effect when parameters change WITH DEBOUNCE"""
        if self._debaunce_job:
            self._params_frame.after_cancel(self._debaunce_job)
        self._debaunce_job = self._params_frame.after(300, self._apply_bilateral_filter)

    #--- Sobel Edge Detection Effect ---
    def _activate_sobel(self):
        """Activate Sobel edge detection effect"""
        self._clear_params()
        tk.Label(self._params_frame, text="Blur Kernel Size:").pack(anchor="w")
        self._blur_ksize_var = tk.IntVar(value=5)
        slider = tk.Scale(
            self._params_frame,
            from_=1, to=99,
            resolution=2,
            orient="horizontal",
            variable=self._blur_ksize_var,
            command=self._on_sobel_param_changed,
        )
        slider.pack(fill="x", pady=(0,6))

        self._apply_sobel()  # Apply effect immediately

    def _apply_sobel(self):
        """Apply Sobel edge detection effect to the image"""
        blur_ksize = self._blur_ksize_var.get()
        self.bgr = sobel_edge_detection(self.original_bgr, blur_ksize=blur_ksize)  # Use the original image
        self._refresh_image()

    def _on_sobel_param_changed(self, _event = None):
        """Re-apply Sobel edge detection effect when parameters change"""
        self._apply_sobel()


    #--- Canny Edge Detection Effect ---
    def _activate_canny(self):
        """Activate Canny edge detection effect and show parameters"""
        self._clear_params()
        tk.Label(self._params_frame, text="Blur Kernel Size:").pack(anchor="w")
        self._canny_blur_ksize_var = tk.IntVar(value=5)
        slider_blur = tk.Scale(
            self._params_frame,
            from_=1, to=99,
            resolution=2,
            orient="horizontal",
            variable=self._canny_blur_ksize_var,
            command=self._on_canny_param_changed,
        )
        slider_blur.pack(fill="x", pady=(0,6))

        tk.Label(self._params_frame, text="Low Threshold:").pack(anchor="w")
        self._canny_low_threshold_var = tk.IntVar(value=50)
        slider_low = tk.Scale(
            self._params_frame,
            from_=0, to=255,
            orient="horizontal",
            variable=self._canny_low_threshold_var,
            command=self._on_canny_param_changed,
        )
        slider_low.pack(fill="x", pady=(0,6))

        tk.Label(self._params_frame, text="High Threshold:").pack(anchor="w")
        self._canny_high_threshold_var = tk.IntVar(value=150)
        slider_high = tk.Scale(
            self._params_frame,
            from_=0, to=255,
            orient="horizontal",
            variable=self._canny_high_threshold_var,
            command=self._on_canny_param_changed,
        )
        slider_high.pack(fill="x", pady=(0,6))

        self._apply_canny()  # Apply effect immediately

    def _apply_canny(self):
        """Apply Canny edge detection effect to the image"""
        blur_ksize = self._canny_blur_ksize_var.get()
        low_threshold = self._canny_low_threshold_var.get()
        high_threshold = self._canny_high_threshold_var.get()
        self.bgr = canny_edge_detection(self.original_bgr, blur_ksize=blur_ksize, low_threshold=low_threshold, high_threshold=high_threshold)  # Use the original image
        self._refresh_image()

    def _on_canny_param_changed(self, _event = None):
        """Re-apply Canny edge detection effect when parameters change"""
        self._apply_canny()

    #--- Black & White Effect ---
    def _activate_black_and_white(self):
        """Activate black & white effect"""
        self._clear_params()
        self._apply_black_and_white()  

    def _apply_black_and_white(self):
        """Apply black & white effect to the image"""
        self.bgr = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2GRAY)  
        self._refresh_image()

    #--- Sepia Effect ---
    def _activate_sepia(self):
        """Activate sepia effect"""
        self._clear_params()
        self._apply_sepia()

    def _apply_sepia(self):
        """Apply sepia effect to the image"""
        self.bgr = sepia_filter(self.original_bgr)
        self._refresh_image()

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

    #--- Color Pencil Sketch Effect ---
    def _activate_color_pencil_sketch(self):
        """Activate color pencil sketch effect and show parameters"""
        self._clear_params()
        tk.Label(self._params_frame, text="Blur Kernel Size:").pack(anchor="w")
        self._kernel_var = tk.IntVar(value=21)
        slider = tk.Scale(
            self._params_frame,
            from_=1, to=99,
            resolution=2,
            orient="horizontal",
            variable=self._kernel_var,
            command=self._on_color_pencil_sketch_param_changed,
        )
        slider.pack(fill="x", pady=(0,6))
        self._apply_color_pencil_sketch()  

    def _apply_color_pencil_sketch(self):
        """Apply color pencil sketch effect to the image"""
        kernel = self._kernel_var.get()
        self.bgr = color_pencil_sketch(self.original_bgr, blur_ksize=kernel)  
        self._refresh_image()

    def _on_color_pencil_sketch_param_changed(self, _event = None):
        """Re-apply color pencil sketch effect when parameters change"""
        self._apply_color_pencil_sketch()



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
        
    def _compose_image(self):
        """Applies the current effect to the original image and updates the editor"""
        # Check if the image is grayscale (pencil sketch) and convert back to BGR if needed
        if len(self.bgr.shape) == 2:  # Grayscale image
            self.bgr = cv2.cvtColor(self.bgr, cv2.COLOR_GRAY2BGR)
        self.original_bgr = self.bgr.copy()  
        self._reset_image()  # Refresh the display with the new original image

    def _save_png(self):
        """Save the current image as a PNG file"""
        path = filedialog.asksaveasfilename(
            title="Save image as",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if path:
            cv2.imwrite(path, self.bgr)
    def close(self):
        """Handle closing the editor"""
        self.destroy()
        if self.on_close:
            self.on_close()
    