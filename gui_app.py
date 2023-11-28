import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from main import main

class MyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Script Interface")

        # Create a text widget for terminal-like output
        self.output_text = tk.Text(root, height=10, width=50)
        self.output_text.pack(padx=10, pady=10)

        # Create a frame for Matplotlib plot
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(padx=10, pady=10)

        # Create a Matplotlib figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.plot = self.figure.add_subplot(1, 1, 1)

        # Create a canvas to display Matplotlib plot in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Create a button to trigger the script
        self.run_button = ttk.Button(root, text="Run Script", command=self.run_script)
        self.run_button.pack(pady=10)

    def run_script(self):
        # Your script logic goes here
        # For demonstration, let's print something to the output_text and plot a simple graph
        main()

def gui_main():
    # Your main script logic goes here
    # main()
    # Example: Print a message
    print("Hello from the main function!")

    # Example: Create an instance of MyApp and run the Tkinter main loop
    root = tk.Tk()
    app = MyApp(root)
    root.mainloop()

if __name__ == "__main__":
    gui_main()
