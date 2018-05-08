import os

SECRET_KEY = os.environ.get('SECRET_KEY') or '#d#JCqTTW\nilK\\7m\x0bp#\tj~#G'

# Database initialization
BASEDIR = os.path.abspath(os.path.dirname(__file__))
BASESAVE = BASEDIR + "/autotag/save/"
