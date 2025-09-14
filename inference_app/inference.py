import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from pathlib import Path
from img_proprocess import preprocess_image, TARGET_HEIGHT, TARGET_WIDTH
from decoder import setup_decoder, wbs_decode
from NN import CRNN, ResidualBlock, MultiHeadSpatialAttention
from PIL import Image, ImageTk
import numpy as np
import threading
import os

class OCRApp:
    def __init__(self, lm_path, nn_path, vocab):
        self.lm_path = lm_path
        self.nn_path = nn_path
        self.vocab = vocab
        self.wbs_decoder = None
        self.model = None
        self.model_loaded = False
        
        # Create main window
        self.root = TkinterDnD.Tk()
        self.root.title("OCR Text Recognition")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        self.setup_ui()
        self.load_models_async()
        
    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root, 
            text="OCR Text Recognition", 
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=10)
        
        # Model loading status
        self.status_label = tk.Label(
            self.root,
            text=" Loading models...",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#666'
        )
        self.status_label.pack(pady=5)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left panel for image
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side='left', expand=True, fill='both', padx=(0, 10))
        
        # Image display area with drag and drop
        self.image_frame = tk.Frame(
            left_frame,
            relief='ridge',
            borderwidth=2,
            bg='white'
        )
        self.image_frame.pack(expand=True, fill='both', pady=(0, 10))
        
        # Drag and drop label
        self.drop_label = tk.Label(
            self.image_frame,
            text=" Drag & Drop Image Here\nor Click Browse Button",
            font=("Arial", 12),
            bg='white',
            fg='#888',
            justify='center'
        )
        self.drop_label.pack(expand=True)
        
        # Enable drag and drop
        self.image_frame.drop_target_register(DND_FILES)
        self.image_frame.dnd_bind('<<Drop>>', self.on_drop)
        
        # Image label for displaying images
        self.image_label = tk.Label(self.image_frame, bg='white')
        
        # Browse button
        self.browse_btn = tk.Button(
            left_frame,
            text=" Browse Image",
            command=self.browse_file,
            font=("Arial", 11),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.browse_btn.pack(pady=5)
        
        # Right panel for results
        right_frame = tk.Frame(main_frame, bg='#f0f0f0', width=300)
        right_frame.pack(side='right', fill='both', padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Results label
        results_label = tk.Label(
            right_frame,
            text="Recognition Results:",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        results_label.pack(anchor='w', pady=(0, 5))
        
        # Text output area
        self.text_output = tk.Text(
            right_frame,
            height=15,
            width=35,
            wrap='word',
            font=("Arial", 10),
            bg='white',
            fg='#333',
            relief='sunken',
            borderwidth=2
        )
        self.text_output.pack(expand=True, fill='both', pady=(0, 10))
        
        # Scrollbar for text output
        scrollbar = tk.Scrollbar(self.text_output)
        scrollbar.pack(side='right', fill='y')
        self.text_output.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.text_output.yview)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            right_frame,
            mode='indeterminate',
            length=250
        )
        
        # Copy button
        self.copy_btn = tk.Button(
            right_frame,
            text=" Copy Text",
            command=self.copy_text,
            font=("Arial", 10),
            bg='#2196F3',
            fg='white',
            padx=15,
            pady=5,
            cursor='hand2',
            state='disabled'
        )
        self.copy_btn.pack(pady=5)
        
        # Current file label
        self.file_label = tk.Label(
            self.root,
            text="No file selected",
            font=("Arial", 9),
            bg='#f0f0f0',
            fg='#666'
        )
        self.file_label.pack(pady=5)
        
    def load_models_async(self):
        """Load models in a separate thread to avoid blocking the UI"""
        threading.Thread(target=self.load_models, daemon=True).start()
        
    def load_models(self):
        try:
            # Update status
            self.root.after(0, lambda: self.status_label.config(text=" Loading language model..."))
            
            # Load decoder
            self.wbs_decoder = setup_decoder(self.vocab, self.lm_path)
            
            # Update status
            self.root.after(0, lambda: self.status_label.config(text=" Loading neural network..."))
            
            # Load model
            checkpoint = torch.load(self.nn_path, map_location="cpu")
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model = CRNN(
                    vocab_size=len(self.vocab),
                    hidden_size=512,
                    num_lstm_layers=2,
                    dropout=0.3,
                    use_attention=True
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model = checkpoint
            
            self.model.eval()
            self.model_loaded = True
            
            # Update status
            self.root.after(0, lambda: self.status_label.config(text=" Models loaded successfully!", fg='green'))
            
        except Exception as e:
            error_msg = f"Error loading models: {str(e)}"
            self.root.after(0, lambda: self.status_label.config(text=error_msg, fg='red'))
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
    
    def on_drop(self, event):
        """Handle drag and drop events"""
        files = event.data.split()
        if files:
            file_path = files[0].strip('{}')  # Remove curly braces if present
            self.process_image(file_path)
    
    def browse_file(self):
        """Open file dialog to browse for image"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Models are still loading. Please wait...")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.process_image(file_path)
    
    def process_image(self, file_path):
        """Process the selected image"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Models are still loading. Please wait...")
            return
            
        try:
            # Validate file exists and is an image
            if not os.path.exists(file_path):
                messagebox.showerror("Error", "File does not exist!")
                return
                
            # Display image
            self.display_image(file_path)
            
            # Update file label
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"Processing: {filename}")
            
            # Show progress bar
            self.progress.pack(pady=5)
            self.progress.start()
            
            # Run OCR in separate thread
            threading.Thread(
                target=self.run_ocr,
                args=(file_path,),
                daemon=True
            ).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")
    
    def display_image(self, file_path):
        """Display the image in the GUI"""
        try:
            # Open and resize image for display
            pil_img = Image.open(file_path)
            
            # Calculate display size (max 400x300, maintain aspect ratio)
            display_width = 400
            display_height = 300
            img_ratio = pil_img.width / pil_img.height
            
            if img_ratio > display_width / display_height:
                new_width = display_width
                new_height = int(display_width / img_ratio)
            else:
                new_height = display_height
                new_width = int(display_height * img_ratio)
            
            display_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_img)
            
            # Hide drop label and show image
            self.drop_label.pack_forget()
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
            self.image_label.pack(expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image:\n{str(e)}")
    
    def run_ocr(self, file_path):
        """Run OCR prediction in background thread"""
        try:
            # Predict text
            predicted_text = self.predict_text(file_path)
            
            # Update UI in main thread
            self.root.after(0, self.update_results, predicted_text, file_path)
            
        except Exception as e:
            error_msg = f"OCR Error: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("OCR Error", error_msg))
            self.root.after(0, self.hide_progress)
    
    def predict_text(self, image_path):
        """OCR prediction function"""
        pil_img = Image.open(image_path)
        img = preprocess_image(
            pil_img, TARGET_HEIGHT, TARGET_WIDTH, True, "0_1", True, "adaptive_gaussian"
        )
        
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:
                img = img.unsqueeze(0)
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError("preprocess_image must return numpy array or torch.Tensor")
        
        with torch.no_grad():
            out = self.model(img)
        
        pred = wbs_decode(out, self.wbs_decoder, blank_is_first=True, chars=self.vocab)[0]
        return pred
    
    def update_results(self, text, file_path):
        """Update the results in the UI"""
        # Clear previous text
        self.text_output.delete(1.0, tk.END)
        
        # Insert new text
        self.text_output.insert(tk.END, text)
        
        # Update file label
        filename = os.path.basename(file_path)
        self.file_label.config(text=f"Processed: {filename}")
        
        # Enable copy button
        self.copy_btn.config(state='normal')
        
        # Hide progress bar
        self.hide_progress()
    
    def hide_progress(self):
        """Hide the progress bar"""
        self.progress.stop()
        self.progress.pack_forget()
    
    def copy_text(self):
        """Copy text to clipboard"""
        text = self.text_output.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Copied", "Text copied to clipboard!")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

# === MAIN APPLICATION ===
if __name__ == "__main__":
    # === CONFIG ===
    LM_PATH = "/home/tass/ocr/ocr_app/lm_model/kenlm_model_3gram.arpa"
    NN_PATH = "/home/tass/ocr/ocr_app/NN/crnn_final.pth"
    
    # Vocab must match what you trained on
    vocab = ['<BLANK>'] + list("!#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")
    
    # Create and run the app
    app = OCRApp(LM_PATH, NN_PATH, vocab)
    app.run()
