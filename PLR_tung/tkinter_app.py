import tkinter as tk
from rembg import remove
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from load_predict import predict
path = ""
root = tk.Tk()
root.geometry("600x600")
root.title("Garbage Classification")
output = 'racc_output.png'
labels = tk.Label(root, font=("Time New Roman", 16))
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        try:
            output = Image.open(file_path)
            # output = remove(output)
            output = output.resize((300, 300), Image.ANTIALIAS)
            output = ImageTk.PhotoImage(output)
            label.config(image=output)
            label.image = output
        except:
            messagebox.showerror("Error", "The file is not an image.")

    img = Image.open(file_path)
    name = predict(img)
    labels.config(text=name)
    labels.pack()
def clear():
    labels.config(text = "")
    labels.pack()
label = tk.Label(root, text="Click the button to open an image.", font=("Time New Roman", 16))
label.pack(pady=10)
bt_clear = tk.Button(root,text="clear", command=clear)
open_file_button = tk.Button(root, text="Open Image", command=open_file,font=("Time New Roman", 16))
open_file_button.pack(pady=10)

root.mainloop()