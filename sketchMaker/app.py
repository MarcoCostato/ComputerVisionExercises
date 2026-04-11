import tkinter as tk
from views.editor_view import EditorView
from views.select_view import SelectView

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("sketchMaker")
        self.geometry("800x200")
        self.resizable(False, False)

        # Show image selection view
        self._show_select_view()

    def _show_select_view(self):
        view = SelectView(self, on_open=self._on_image_selected)
        view.pack(fill="both", expand=True, padx=20, pady=20)

    def _on_image_selected(self, image_path):
        self.withdraw()  # Hide main window
        EditorView(self, image_path, on_close=self._on_editor_closed)

    def _on_editor_closed(self):
        self.deiconify()  # Show main window again