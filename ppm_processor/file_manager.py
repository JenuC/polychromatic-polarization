import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil

class FileOrganizer(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        master.title("File Organizer")
        self.pack(fill="both", expand=True)
        self.create_widgets()

    def create_widgets(self):
      
        self.drop_target = tk.LabelFrame(self, text="Drag files/folders here", width=300, height=200)
        self.drop_target.pack(pady=20)
        self.drop_target.pack_propagate(False)

        self.drop_target.bind("<Enter>", self.drag_enter)
        self.drop_target.bind("<Leave>", self.drag_leave)
        self.drop_target.bind("<B1-Motion>", self.drag_motion)
        self.drop_target.bind("<ButtonRelease-1>", self.drop)
        self.files = []
        
        self.organize_button = tk.Button(self, text="Organize Files", command=self.organize_files)
        self.organize_button.pack(pady=10)

        self.status_label = tk.Label(self, text="")
        self.status_label.pack()

    def drag_enter(self, event):
        self.drop_target.config(bg="lightblue")

    def drag_leave(self, event):
        self.drop_target.config(bg="SystemButtonFace")
    
    def drag_motion(self, event):
        self.drop_target.config(bg="lightblue")

    def drop(self, event):
      self.drop_target.config(bg="SystemButtonFace")
      data = event.data
      if data:
        file_list = data.split()
        self.files.extend(file_list)
        self.status_label.config(text=f"Files added: {', '.join([os.path.basename(f) for f in file_list])}")
      else:
        self.status_label.config(text="No files dropped")

    def organize_files(self):
        if not self.files:
            messagebox.showinfo("Info", "No files to organize.")
            return

        target_dir = filedialog.askdirectory(title="Select Target Directory")
        if not target_dir:
            return

        for file_path in self.files:
            try:
                if os.path.isfile(file_path):
                    file_name = os.path.basename(file_path)
                    extension = file_name.split(".")[-1].lower()
                    dest_folder = os.path.join(target_dir, extension)

                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    
                    shutil.move(file_path, os.path.join(dest_folder, file_name))
                elif os.path.isdir(file_path):
                    folder_name = os.path.basename(file_path)
                    dest_folder = os.path.join(target_dir, folder_name)
                    shutil.move(file_path, dest_folder)

                self.status_label.config(text="Files organized successfully!")
                self.files = []
            except Exception as e:
                messagebox.showerror("Error", f"Error organizing files: {e}")
                return

root = tk.Tk()
app = FileOrganizer(root)
root.mainloop()