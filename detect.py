import cv2
import numpy as np
from PIL import ImageGrab
import torch
import tkinter as tk


def detect():
    width, height = get_current_screen()
    # Model
    model = torch.hub.load('ultralytics/yolov5', "custom", 'yolov5s')
    # model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt',
    #                        force_reload=True, trust_repo=True)

    model.classes = 0
    model.conf = 0.50

    while True:
        w, h = width, height
        monitor = {'top': 0, 'left': 0, 'width': w, 'height': h}
        screen = np.array(ImageGrab.grab())
        screen = cv2.cvtColor(src=screen, code=cv2.COLOR_BGR2RGB)
        # set the model use the screen
        result = model(screen)

        print("Total person:" + str(len(result.xyxy[0])))
        # print(result.xyxy[0])
        # print(result.pandas().xyxy[0])
        cv2.imshow('Screen', result.render(labels=False)[0])

        # click p to quit
        if cv2.waitKey(27) & 0xFF == ord('p'):
            break

    cv2.destroyAllWindows()


def get_current_screen():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height


def myTest():
    pass


if __name__ == "__main__":
    myTest()
    detect()
