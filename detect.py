import cv2
import numpy as np
from PIL import ImageGrab
import torch
import tkinter as tk
import pandas as pd
import pyautogui as screen
import mss
import time
from draw import Draw

root = tk.Tk()
ex = Draw()


def detect():
    with mss.mss() as sct:
        width, height = get_current_screen()
        monitor = {"top": 40, "left": 0, "width": width, "height": height}
        # Model
        model = torch.hub.load('ultralytics/yolov5', "custom", 'yolov5s', "source='local'")
        # model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt',
        #                        force_reload=True, trust_repo=True)

        # self-config
        mid = np.array([width, height])
        model.classes = 0
        model.conf = 0.5

        autoShooting, displayFPS, showDetection = True, True, True

        while True:
            # draw_tkinter()
            # my_screen = np.array(ImageGrab.grab())
            # my_screen = cv2.cvtColor(src=my_screen, code=cv2.COLOR_BGR2RGB)
            last_time = time.time()
            my_screen = np.array(sct.grab(monitor))
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

            if autoShooting & len(temp) != 0:
                x = (df.loc[temp]['xmax'] + df.loc[temp]['xmin']) / 4
                y = (df.loc[temp]['ymax'] + df.loc[temp]['ymin']) / 4
                lengthOfPerson = ((df.loc[temp]['ymax']/4) - (df.loc[temp]['ymin']/4)) * 0.15
                # y = (df.loc[temp]['ymax'] + df.loc[temp]['ymin']) / 4
                mouse_move_and_click(x, y + lengthOfPerson)
            # print(df.loc[x]['xmin'])

            # print(df[df['dist'] == df['dist'].min()].index.values)
            # print(df)
            # closest_row = df[df["distance"] == df["distance"].min()]
            # print(closest_row)


            # cv2.putText(result.render(labels=False)[0],
            #             "FPS: {}".format(1 / (time.time() - last_time)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            if displayFPS:
                print("FPS: {}".format(1 / (time.time() - last_time)))

            if showDetection:
                cv2.imshow('Detection', result.render(labels=False)[0])

            # click l to quit
            key = cv2.waitKey(1)
            if key & 0xFF == ord("l"):
                break

        # out.release()
        cv2.destroyAllWindows()
        print("Exit")


def get_current_screen():
    # root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    # root.destroy()
    return screen_width, screen_height


def draw_tkinter():
    root.geometry("420x250+0+0")
    root.attributes('-alpha', 0.5)
    root.after(2000, detect)
    root.mainloop()


def mouse_move_and_click(x, y):
    x = int(x)
    y = int(y)
    screen.moveTo(x, y)
    screen.click(x, y)
    # time.sleep(10)


if __name__ == "__main__":
    # draw_tkinter()
    # myTest()
    detect()
