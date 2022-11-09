from flask import Flask, render_template, request
from pandas import read_csv
from patsy import PatsyError

from config import general_config
from ml_core.expert_score_regressor.estimator import ExpertScoreModelEstimator
from ml_core.weak_link_classifier.estimator import WeakLinkModelEstimator

app = Flask(__name__)
# Restrict large files >5MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

ALLOWED_EXTENSIONS = ["csv"]
aimoEst = ExpertScoreModelEstimator()
weaklink_est = WeakLinkModelEstimator()


@app.route("/", methods=["GET"])
def index():
    return render_template("home.html")


@app.route("/eval_score", methods=["GET"])
def eval():
    return render_template("eval_score.html")


@app.route("/eval_weaklink", methods=["GET"])
def eval_weaklink():
    return render_template("eval_weaklink.html")


def evaluate_score(file):
    df = read_csv(file, decimal=',')
    try:
        aimo_score = aimoEst.eval(df)[0]
    except PatsyError:
        return render_template("eval_score.html",
                               msg="Error: The file with the NASM and Angle features should be uploaded")
    aimo_score = round(aimo_score, 3)
    return render_template("eval_score.html", score_pred=aimo_score)


def evaluate_weaklink(file):
    df = read_csv(file, decimal=',')
    try:
        weaklink = weaklink_est.eval(df)[0]
    except PatsyError:
        return render_template("eval_weaklink.html",
                               msg="Error: The file with the NASM and Angle features should be uploaded")
    return render_template("eval_weaklink.html", weaklink_pred=weaklink)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[
        1].lower() in ALLOWED_EXTENSIONS


@app.route("/eval_score", methods=["POST"])
def submit_score():
    request_file = request.files['data_file']
    if not request_file:
        return render_template("eval_score.html", msg="Please select a file")
    if not allowed_file(request_file.filename):
        return render_template("eval_score.html",
                               msg="Error: You can only upload " + str(
                                   ALLOWED_EXTENSIONS).strip('[]'))
    return evaluate_score(request_file)


@app.route("/eval_weaklink", methods=["POST"])
def submit_weaklink():
    request_file = request.files['data_file']
    if not request_file:
        return render_template("eval_weaklink.html", msg="Please select a file")
    if not allowed_file(request_file.filename):
        return render_template("eval_weaklink.html",
                               msg="Error: You can only upload " + str(
                                   ALLOWED_EXTENSIONS).strip('[]'))
    return evaluate_weaklink(request_file)


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == "__main__":
    app.run(general_config.service.host, general_config.service.port)
