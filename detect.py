import cv2
import numpy as np
from PIL import ImageGrab
import torch
import tkinter as tk
import pandas as pd
import pyautogui as screen
import time

from draw import Draw


def detect():
    width, height = get_current_screen()
    # Model
    model = torch.hub.load('ultralytics/yolov5', "custom", 'yolov5s')
    # model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt',
    #                        force_reload=True, trust_repo=True)

    mid = np.array([width, height])
    # print(str(mid[0]) + ", " + str(mid[1]))
    model.classes = 0
    model.conf = 0.1
    model.imgsz = (height, width)

    while True:
        # draw_tkinter()
        my_screen = np.array(ImageGrab.grab())
        my_screen = cv2.cvtColor(src=my_screen, code=cv2.COLOR_BGR2RGB)

        # cv2.rectangle(my_screen, (624, 635), (1070, 967), (0, 255, 0), 3)

        result = model(my_screen)

        # print("Total person:" + str(len(result.xyxy[0])))
        # print(result.pandas().xyxy[0])

        df = result.pandas().xyxy[0]
        # print("test")
        # df["distance"] = (df["xmin"] - width) ** 2 + (df["ymin"] - height) ** 2
        # closest_row = df[df["distance"] == df["distance"].min()]

        df['dist'] = ((df['xmin'] - mid[0]) ** 2 + (df['ymin'] - mid[1]) ** 2).apply(np.sqrt)
        temp = df[df['dist'] == df['dist'].min()].index.values

        if len(temp) != 0:
            x = (df.loc[temp]['xmax'] + df.loc[temp]['xmin'])/4
            y = (df.loc[temp]['ymax'] + df.loc[temp]['ymin'])/4.5
            mouse_move_and_click(x, y + 10)
        # print(df.loc[x]['xmin'])
        # if len(x) != 0:
        #     print(x[0])

        # print(df[df['dist'] == df['dist'].min()].index.values)
        # print(df)
        # closest_row = df[df["distance"] == df["distance"].min()]
        # print(closest_row)

        cv2.imshow('Detection', result.render(labels=False)[0])
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


def draw_tkinter():
    root = tk.Tk()
    ex = Draw()
    root.geometry("420x250+0+0")
    # root.mainloop()


def mouse_move_and_click(x, y):
    x = int(x)
    y = int(y)
    screen.moveTo(x, y)
    screen.click(x, y)
    time.sleep(1000)


if __name__ == "__main__":
    # draw_tkinter()
    # myTest()
    detect()