import pandas as pd
import os
import datetime
import time
from flask import Flask, redirect, url_for

async def createTestFile(fullPath):
    print("====inside createTestFile func====")
    filename = os.path.basename(fullPath)
    filename_no_ext = os.path.splitext(filename)[0]
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    filename = "{}_{}.xlsx".format(filename_no_ext, datetime.datetime.now().timestamp())
    if "CannotWork" in filename_no_ext: 
        raise Exception('this file cannot work')
    filePath = "processed/" + filename
    df.to_excel(filePath)
    for x in range(3):
        time.sleep(1)
        if(x%50 == 0):
            print("\t***", x)
    else:
        print("\t***Finally finished!")
    return filename
    