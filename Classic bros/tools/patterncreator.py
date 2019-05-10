import tkinter as tk
import numpy as np
import sys
from functools import partial

class Window(tk.Frame):

    def __init__(self,dimension,filename,master=None):
        tk.Frame.__init__(self,master)
        self.master = master
        self.dimension = dimension
        self.patterns = []
        self.currentpattern = np.full(dimension[0]*dimension[1],-1)
        self.buttons = []
        self.filename = filename
        self.init_window()
    
    def clear(self):
        for btn in self.buttons:
            btn.configure(bg='white')
        print(self.currentpattern)
        self.currentpattern = np.full(self.dimension[0]*self.dimension[1],-1)
        print(self.currentpattern)
        print(self.patterns)

    def reset(self):
        self.clear()
        self.patterns = []
    
    def submit(self):
        self.patterns.append(self.currentpattern.copy())
        self.clear()

    def save(self):
        name = self.filename
        to_save = np.array(self.patterns)
        np.save(name,to_save)
        self.reset()
    
    def actionOnPress(self,index):
        widget = self.buttons[index]
        self.currentpattern[index] *= -1
        if widget['bg'] == 'white':
            widget.configure(bg='black')
        else:
            widget.configure(bg='white')

    def init_window(self):
        rows, cols = self.dimension
        self.master.title("Pattern generator")
        self.grid()
        buttonframe = tk.Frame(self.master)
        buttonframe.grid()
        for row in range(rows):
            for col in range(cols):
                buttonfunc = partial(self.actionOnPress, col + cols * row)
                butt = tk.Button(buttonframe, text=int(row*cols + col),height = 1,width=2,bg="white",command = buttonfunc)
                butt.grid(row=row,column=col,sticky=tk.NSEW)
                self.buttons.append(butt)
        resetbutton = tk.Button(self.master,text = "reset",command=self.clear)
        resetbutton.grid()
        submitbutton = tk.Button(self.master,text = "submit",command=self.submit)
        submitbutton.grid()
        savebutton = tk.Button(self.master,text = "save",command=self.save)
        savebutton.grid()
        clearbutton = tk.Button(self.master,text = "clear",command=self.reset)
        clearbutton.grid()



if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("pls specify the rows and cols and target filename")
        print("Usage: patterncreator.py <rows> <cols> <filename>")
        exit()
    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
    root = tk.Tk()
    app = Window(dimension=(rows,cols),filename=sys.argv[3],master=root)
    root.geometry("400x300")
    root.mainloop()
