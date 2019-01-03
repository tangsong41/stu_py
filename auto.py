# -*- coding: utf-8 -*-
import os
import psutil
from pykeyboard import PyKeyboard
from pywinauto import application

def close_process(process_name):
    if process_name[-4:].lower() != ".exe":
        process_name += ".exe"
    os.system("taskkill /f /im " + process_name)

def upload(process_id,filepath):
    app = application.Application()
    app.connect(process=int(process_id))
    window = app.window_(class_name="#32770")
    window["Edit"].TypeKeys(filepath)
    press_keyboard('Enter',1)
    window[u"打开(O)"].Click()

def process_id(process_name):
    process_list = list(psutil.process_iter())
    pid = []
    for process in process_list:
        if 'name=\''+process_name+'\'' in str(process):
            print process
            id= str(process).split('(')[1].split(',')[0].split('=')[1]
            pid.append(id)
    return pid

def press_keyboard(keyname,nums):
    k = PyKeyboard()	
    nums= int(nums)
    print keyname
    if keyname == "Tab":
        k.tap_key(k.tab_key,n=nums,interval=1)
    elif keyname == "Space":
        k.tap_key(k.space_key,n=nums,interval=1)
    elif keyname == "Enter":
        k.tap_key(k.enter_key,n=nums,interval=1)
    else:
        k.tap_key(keyname,n=nums,interval=1)

def maximize(process_id):
    app = application.Application()
    app.connect(process=int(process_id))
    window = app.window_(class_name="IEFrame")
    window.Maximize()
    window.click()

