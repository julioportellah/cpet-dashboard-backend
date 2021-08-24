import custom_shap as cshap
import pandas as pd
from flask import jsonify

class PatientFullPrediction():
    def __init__(self, session_id, patient_id, cardiac_proba, cardiac_lim, pulmonary_proba,
                 pulmonary_lim, other_proba, other_lim) -> None:
        self.session_id = int(session_id)
        self.patient_id = float(patient_id)
        self.cardiac_proba = int(cardiac_proba*100)
        self.cardiac_lim = int(cardiac_lim*100)
        self.pulmonary_proba = int(pulmonary_proba*100)
        self.pulmonary_lim = int(pulmonary_lim*100)
        self.other_proba = int(other_proba*100)
        self.other_lim = int(other_lim*100)
        pass
    pass

class PatientDynamicFullPrediction():
    def __init__(self, session_id, patient_id, cardiac_proba_array, cardiac_lim, pulmonary_proba_array,
                 pulmonary_lim, other_proba_array, other_lim) -> None:
        self.session_id = int(session_id)
        self.patient_id = float(patient_id)
        self.cardiac_lim = int(cardiac_lim*100)
        self.pulmonary_lim = int(pulmonary_lim*100)
        self.other_lim = int(other_lim*100)
        self.cardiac_proba = cardiac_proba_array
        self.pulmonary_proba = pulmonary_proba_array
        self.other_proba = other_proba_array
        self.time_list = [40, 50, 60, 70, 80, 90, 100]
        pass
    pass


def _generate_array(df, type_lim):
    result = []
    time_list = [40, 50, 60, 70, 80, 90, 100]
    for time in time_list:
        result.append(int(df[type_lim+'LimProba_'+str(time)].values[0]*100))
    return result
    pass

def get_dynamic_cpet_record_by_session_id(session_id):
    """API that retrieves the dynamic records of a patient

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
        session_id = float(session_id)
        data_df = pd.read_csv('.\\data\\data_export_dynamic.csv')
        data_filtered = data_df.loc[data_df.SessionId == session_id]
        cardiac_array = _generate_array(data_filtered, 'Cardiac')
        pulmonary_array = _generate_array(data_filtered, 'Pulmonary')
        other_array = _generate_array(data_filtered, 'Other')
        result = PatientDynamicFullPrediction(data_filtered.SessionId.values[0], data_filtered.PatientId.values[0], 
                                cardiac_array, data_filtered.CardiacLim.values[0],
                                pulmonary_array, data_filtered.PulmonaryLim.values[0],
                                other_array, data_filtered.OtherLim.values[0])
        return result, 200
    except Exception as e:
        print(e)
        return "Unexpected error", 400
    pass

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
        session_id = float(session_id)
        data_df = pd.read_csv('.\\data\\cpet_full_proba.csv')
        data_filtered = data_df.loc[data_df.SessionId == session_id]
        #print(data_filtered.SessionId.values[0])
        result = PatientFullPrediction(data_filtered.SessionId.values[0], data_filtered.PatientId.values[0], 
                                data_filtered.CardiacLimProba.values[0], data_filtered.CardiacLim.values[0],
                                data_filtered.PulmonaryProba.values[0], data_filtered.PulmonaryLim.values[0],
                                data_filtered.OtherProba.values[0], data_filtered.OtherLim.values[0])
        #print(result.cardiac_proba)
        #res, code = hr.get_record_by_patient_id(patient_id)
        return result, 200
    except Exception as e:
        #logging.error(traceback.format_exc())
        print(e)
        return "Unexpected error", 400
    pass


if __name__ == "__main__":
    #session_id = float("78.2")
    #data_df = pd.read_csv('.\\data\\cpet_full_proba.csv')
    print(get_dynamic_cpet_record_by_session_id("78.2"))
    #print(get_cpet_record_by_session_id("78.2"))
    pass

