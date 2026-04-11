import tkinter as tk
from tkinter import filedialog
import os

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

class SelectView(tk.Frame):
    def __init__(self, parent, on_open):
        super().__init__(parent)
        self.on_open = on_open

        self._path_var = tk.StringVar()
        self._error_var = tk.StringVar()
        self._build()
        
    def _build(self):
        # Instruction label
        tk.Label(self, text="Select an image:").pack(anchor="w")

        # Row: Entry + Browse button
        row = tk.Frame(self)
        row.pack(fill="x", pady=(6,0))

        self._entry = tk.Entry(row, textvariable=self._path_var)
        self._entry.pack(side="left", fill="x", expand=True)

        tk.Button(row, text="Browse", command=self._browse).pack(side="left", padx=(6,0))
        # Error message
        self._error_label = tk.Label(self, textvariable=self._error_var, fg="red")
        self._error_label.pack(anchor="w", pady=(4,0))

        # Open button
        tk.Button(self, text="Open", command=self._open).pack(pady=(12,0))

    def _browse(self):
        """Open file dialog to select an image"""
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                       ("All files", "*.*")
                       ]
        )
        if path:
            self._path_var.set(path)
            self._error_var.set("")  # Clear any previous error

    def _open(self):
        """Validate and open the selected image"""
        path = self._path_var.get().strip()
        # Empty path check
        if not path:
            self._error_var.set("Please select an image.")
            return
        
        # File existence check
        if not os.path.isfile(path):
            self._error_var.set("File does not exist.")
            return
        
        # Extension check
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            self._error_var.set("Unsupported file type ({ext}). Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
            return
        
        # If all checks pass, call the on_open callback
        self._error_var.set("")  # Clear any previous error
        self.on_open(path)