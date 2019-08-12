from flask import Flask
from api.routes import mod

app = Flask(__name__)

app.register_blueprint(routes.mod, url_prefix='/api')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)