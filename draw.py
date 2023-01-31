from tkinter import Canvas, Frame, BOTH, W


class Draw(Frame):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.master.title("Plug In")
        self.pack(fill=BOTH, expand=1)

        canvas = Canvas(self)
        canvas.create_text(10, 10, text='Switch: On', font=('Arial', 20), anchor='nw')
        canvas.pack(fill=BOTH, expand=1)