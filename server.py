from flask import Flask, jsonify, Request
from flask_cors import CORS
import os
import current_patient_service as cps
import json

app = Flask(__name__)
CORS(app)


@app.route("/api/get_record_by_patient_id/<session_id>",
           methods=["GET"])
def get_cpet_record_by_session_id(session_id):
    """API that retrieves all the cpet reocrds of a session

    This method validates the input and calls a method that will retrieve
    a list of all the health records of a patient, if there is one exists.
    The result will be a list of dictionaries containing the information

    Args:
        session_id (str): The id of the health record

    Returns
        `list` of `dict`, int: All patient's medical records, 200
        str, int: Error Message, 400
    """
    try:
        print(session_id)
        result = cps.get_cpet_record_by_session_id(session_id)
        #res, code = hr.get_record_by_patient_id(patient_id)
        return json.dumps(vars(result[0])), 200
    except Exception as e:
        #logging.error(traceback.format_exc())
        print(e)
        return "Unexpected error", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0")


