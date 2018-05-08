# -*- coding: utf-8 -*-
from autotag import app
import logging
from logging.handlers import RotatingFileHandler

if __name__ == "__main__":
    # initialize the log handler
    logHandler = RotatingFileHandler('info.log', maxBytes=100000, backupCount=1)
    # set the log handler level
    logHandler.setLevel(logging.INFO)
    # set the app logger level
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(logHandler)    

    app.run(host='127.0.0.3', port=5004, debug=True)
