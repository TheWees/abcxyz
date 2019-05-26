import os
import pandas as pd
import stub as stub
import Sales_Commentaries_Generator as scg
from flask import Flask, Response, request, render_template, send_from_directory, send_file
import requests
import asyncio
from flask_socketio import SocketIO, emit

__author__ = 'ibininja'

app = Flask(__name__)
socketio = SocketIO(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/original/'
PROCESSED_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/processed/'
print("ORIGINAL_FOLDER", ORIGINAL_FOLDER)
print("PROCESSED_FOLDER", PROCESSED_FOLDER)


@app.route("/")
def index():
    return render_template("upload.html")

@socketio.on('connect')
def test_connect():
    emit('after connect',  {'data':'Lets dance'})

@app.route("/upload", methods=["POST"])
def upload():
    print("====inside upload func====")
    # this is to verify that folder to upload to exists.
    if os.path.isdir(ORIGINAL_FOLDER):
        print("original/ folder exist")
    else:
        print("original/ folder doesn't exist")

    if not os.path.isdir(ORIGINAL_FOLDER):
        os.mkdir(ORIGINAL_FOLDER)
    if not os.path.isdir(PROCESSED_FOLDER):
        os.mkdir(PROCESSED_FOLDER)

    for upload in request.files.getlist("file"):
        print(upload)
        # print("***file name is {}".format(upload.filename))
        filename = upload.filename
        ext = os.path.splitext(filename)[1]
        if (ext == ".xls") or (ext == ".xlsx"):
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")
        destination = "/".join([ORIGINAL_FOLDER, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)
        socketio.emit('processing',  {})

        newFilePath = asyncio.run(getProcessedData(filename))
        return Response(newFilePath)
        # newFilePath = stub.createTestFile(filename_no_ext)
        #return render_template("complete.html", image_name=filename)

async def getProcessedData(filename):
    try:
        fullPath = ORIGINAL_FOLDER + filename
        newFilePath = await scg.generate_sales_comments(file_name=fullPath)
        #newFilePath = await stub.createTestFile(fullPath)
    except Exception as e:
        print("***error in processing file", e)
        return None
    print("***done processing")
    return newFilePath

@app.route("/download/<path:filename>", methods=['GET'])
def download(filename):
    print("====inside download func====")
    # filename = request.args.get('filename')
    return send_from_directory(PROCESSED_FOLDER, filename, 
        as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == "__main__":
    app.run(port=5000, debug=True)
