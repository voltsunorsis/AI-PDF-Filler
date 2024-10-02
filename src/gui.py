import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from testingcode import process_pdf  # Import the process_pdf function

class PDFProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI PDF Filler")
        self.root.geometry("600x400")
        self.create_widgets()

    def create_widgets(self):
        padding = {'padx': 10, 'pady': 10}

        # Input PDF Selection
        self.pdf_label = tk.Label(self.root, text="Select PDF File:")
        self.pdf_label.grid(row=0, column=0, sticky='e', **padding)

        self.pdf_path = tk.StringVar()
        self.pdf_entry = tk.Entry(self.root, textvariable=self.pdf_path, width=50)
        self.pdf_entry.grid(row=0, column=1, **padding)

        self.pdf_button = tk.Button(self.root, text="Browse", command=self.browse_pdf)
        self.pdf_button.grid(row=0, column=2, **padding)

        # Context File Selection
        self.context_label = tk.Label(self.root, text="Select Context File:")
        self.context_label.grid(row=1, column=0, sticky='e', **padding)

        self.context_path = tk.StringVar()
        self.context_entry = tk.Entry(self.root, textvariable=self.context_path, width=50)
        self.context_entry.grid(row=1, column=1, **padding)

        self.context_button = tk.Button(self.root, text="Browse", command=self.browse_context)
        self.context_button.grid(row=1, column=2, **padding)

        # Output Directory Selection
        self.output_label = tk.Label(self.root, text="Select Output Directory:")
        self.output_label.grid(row=2, column=0, sticky='e', **padding)

        self.output_path = tk.StringVar()
        self.output_entry = tk.Entry(self.root, textvariable=self.output_path, width=50)
        self.output_entry.grid(row=2, column=1, **padding)

        self.output_button = tk.Button(self.root, text="Browse", command=self.browse_output)
        self.output_button.grid(row=2, column=2, **padding)

        # Start Button
        self.start_button = tk.Button(self.root, text="Start Processing", command=self.start_processing, bg='green', fg='white')
        self.start_button.grid(row=3, column=1, **padding)

        # Progress Bar
        self.progress = ttk.Progressbar(self.root, orient='horizontal', length=400, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=3, **padding)

        # Status Message
        self.status_message = tk.StringVar()
        self.status_label = tk.Label(self.root, textvariable=self.status_message, fg='blue')
        self.status_label.grid(row=5, column=0, columnspan=3, **padding)

    def browse_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.pdf_path.set(file_path)

    def browse_context(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            self.context_path.set(file_path)

    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_path.set(directory)

    def start_processing(self):
        pdf = self.pdf_path.get()
        context = self.context_path.get()
        output = self.output_path.get()

        if not pdf or not os.path.isfile(pdf):
            messagebox.showerror("Error", "Please select a valid PDF file.")
            return

        if not context or not os.path.isfile(context):
            messagebox.showerror("Error", "Please select a valid context text file.")
            return

        if not output or not os.path.isdir(output):
            messagebox.showerror("Error", "Please select a valid output directory.")
            return

        # Disable the start button to prevent multiple clicks
        self.start_button.config(state='disabled')
        self.status_message.set("Processing started...")
        self.progress['value'] = 0

        # Run the processing in a separate thread to keep the GUI responsive
        threading.Thread(target=self.run_processing, args=(pdf, context, output)).start()

    def run_processing(self, pdf, context, output):
        try:
            # Call the process_pdf function with the selected paths
            process_pdf(pdf, output, context)
            self.progress['value'] = 100
            self.status_message.set("Processing complete!")
            messagebox.showinfo("Success", "PDF processing completed successfully.")
        except Exception as e:
            self.status_message.set("An error occurred.")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.start_button.config(state='normal')

def main():
    root = tk.Tk()
    app = PDFProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()