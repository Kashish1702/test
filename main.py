from flask import Flask


app = Flask(__name__)

@app.route("/")
def hello_world():
    result = sum([3, 4])
    return f"Hello, World! Result of addition: {result}"

if __name__ == "__main__":
    app.run(debug=True)
