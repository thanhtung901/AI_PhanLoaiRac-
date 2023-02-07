import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from load_predict import predict
path = ""


def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        try:
            image = Image.open(file_path)
            image = image.resize((300, 300), Image.ANTIALIAS)
            image = ImageTk.PhotoImage(image)
            label.config(image=image)
            label.image = image
        except:
            messagebox.showerror("Error", "The file is not an image.")

    # print(type(file_path))
    img = Image.open(file_path)
    name = predict(img)
    labels = tk.Label(root, text=name)
    labels.pack()



root = tk.Tk()
root.geometry("600x600")
root.title("Image Upload and Display")

label = tk.Label(root, text="Click the button to open an image.")
label.pack(pady=10)

open_file_button = tk.Button(root, text="Open Image", command=open_file)
open_file_button.pack(pady=10)

root.mainloop()