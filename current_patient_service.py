import custom_shap as cshap
import pandas as pd
from flask import jsonify
import pickle
import shap
from custom_shap import summary_with_highlight 
import uuid
import base64
import os
from PIL import Image as img
from matplotlib import pyplot as plt
import matplotlib.pyplot as pl; 


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


class CPETInterpretationImages():
    def __init__(self, cardiac_summary, cardiac_force, pulmonary_summary, pulmonary_force, 
                other_summary, other_force) -> None:
        self.cardiac_summary = cardiac_summary
        self.cardiac_force = cardiac_force
        self.pulmonary_summary = pulmonary_summary
        self.pulmonary_force = pulmonary_force
        self.other_summary = other_summary
        self.other_force = other_force
        pass


def _generate_array(df, type_lim):
    result = []
    time_list = [40, 50, 60, 70, 80, 90, 100]
    for time in time_list:
        result.append(int(df[type_lim+'LimProba_'+str(time)].values[0]*100))
    return result
    pass


cardiac_data_100 = ['CardiacLim','DiffPercentPeakVO2', 'DiffPeakVO2','75_to_100_VO2Slope','75_to_100_HRSlope','MinO2Pulse',
                      'PeakVE','VO2vsPeakVO2atVT','second_half_RRSlope','second_half_VO2Slope','75_to_100_VCO2Slope','MeanVE',
                      'second_half_VESlope','O2PulseDiff','50_to_75_O2Slope',
                        'O2PulsePercent','75_to_100_RERSlope','PeakRER','50_to_75_VO2Slope','PeakVO2Real']
pulmonary_data_100 = ['PulmonaryLim','O2PulsePercent', 'O2PulseDiff','first_half_VO2Slope','LowestVE/VCO2',
                      'first_half_VCO2Slope', '15_to_85_RRSlope','PeakRR','50_to_75_RRSlope','MeanO2Pulse','VEvsVCO2Slope',
                     '25_to_50_VCO2Slope','StdHeartRate']
other_data_100 = ['MuscleSkeletalLim','PeakRR', 'PeakVE','PeakVCO2','MeanVCO2','PeakVO2','PeakVO2Real',
                  'LowestVE/VCO2','MeanRER','PeakRER','VO2vsPeakVO2atVT','DiffPercentPeakVO2','MeanRR',
                  '75_to_100_VEVCO2Slope','DiffPeakVO2','MeanVE','second_half_VESlope','first_half_VEVCO2Slope',
                  '0_to_25_O2Slope','VO2atVT', 'MeanVO2','second_half_VCO2Slope','DiffPeakHR','MeanVE/VCO2','75_to_100_RRSlope']


def get_interpretation_images_by_id(session_id):
    try:
        data_df= pd.read_csv('.\\data\\data_100.csv')
        session_id = float(session_id)
        image_list_str = []
        lim_types = ['cardiac', 'pulmonary', 'other']
        selected_model = None
        feature_selector = None
        for lim_type in lim_types:
            if lim_type == 'cardiac':
                feature_selector = cardiac_data_100[1:]
            elif lim_type == 'pulmonary':
                feature_selector = pulmonary_data_100[1:]
            else:
                feature_selector = other_data_100[1:]
            shap_values = pickle.load(
                    open(f".\\models\\{lim_type}\\"+lim_type+'_shap_values.sav', 'rb'))
            data_index = data_df.loc[data_df['SessionId'] == session_id].index[0]
            pl_result = summary_with_highlight(shap_values[1], data_df[feature_selector], row_highlight=data_index, max_display=10, as_string=True)
            image_list_str.append(pl_result)
            force_pl_str = create_force_plot_string(lim_type, data_index)
            image_list_str.append(force_pl_str)
        result = CPETInterpretationImages(image_list_str[0],image_list_str[1],image_list_str[2],
                                        image_list_str[3],image_list_str[4],image_list_str[5])
        return result, 200
    except Exception as e:
        print(e)
        return "Unexpected error", 400
    pass


def create_force_plot_string(lim_type, data_index):
    file_name = str(uuid.uuid4())
    file_name_png = file_name + '.png'
    file_name_jpg = file_name + '.jpg'
    feature_selector = None
    if lim_type == 'cardiac':
        feature_selector = cardiac_data_100[1:]
    elif lim_type == 'pulmonary':
        feature_selector = pulmonary_data_100[1:]
    else:
        feature_selector = other_data_100[1:]
    loaded_tree = pickle.load(open(f".\\models\\{lim_type}\\"+lim_type+'_tree_explainer.sav', 'rb'))
    shap_values = pickle.load(open(f".\\models\\{lim_type}\\"+lim_type+'_shap_values.sav', 'rb'))
    shap.force_plot(loaded_tree.expected_value[1], shap_values[1][data_index], feature_names=feature_selector,
            link='identity', contribution_threshold=0.1,show=False, plot_cmap=['#77dd77', '#f99191'],
            matplotlib=True).savefig('.\\temp_images\\'+file_name_png,format = "png",dpi = 150,bbox_inches = 'tight')
    png_img = img.open('.\\temp_images\\'+file_name_png)
    rgb_im = png_img.convert('RGB')
    rgb_im.save('.\\temp_images\\'+file_name_jpg,'JPEG')
    with open('.\\temp_images\\'+file_name_jpg, 'rb') as image_file:
            b64_bytes = base64.b64encode(image_file.read())
    b64_string = str(b64_bytes, encoding='utf-8')
    plt.close()
    os.remove('.\\temp_images\\'+file_name_png)
    os.remove('.\\temp_images\\'+file_name_jpg)
    return b64_string
    pass

def _test_force_plot():
    #shap.initjs()
    f = pl.gcf()
    lim_type = 'cardiac'
    loaded_tree = pickle.load(open(f".\\models\\{lim_type}\\"+lim_type+'_tree_explainer.sav', 'rb'))
    shap_values = pickle.load(open(f".\\models\\{lim_type}\\"+lim_type+'_shap_values.sav', 'rb'))
    shap.force_plot(loaded_tree.expected_value[1], shap_values[-1][0], feature_names=cardiac_data_100[1:],
            link='identity', contribution_threshold=0.1,show=False, plot_cmap=['#77dd77', '#f99191'],
            matplotlib=True).savefig('.\\temp_images\\'+'scratch.png',format = "png",dpi = 150,bbox_inches = 'tight')
    # shap.force_plot(loaded_tree.expected_value[1], shap_values[1][0],  feature_names=cardiac_data_100[1:],
    #                 contribution_threshold=0.9, link='identity', matplotlib=True, show=False)
    # plt.savefig('.\\temp_images\\'+'scratch2.png')
    #pl.savefig('.\\temp_images\\'+'scratch.png')
    #plt.show()
    # pl.tight_layout()
    #pl.savefig('.\\temp_images\\'+file_name_png,dpi=pl.gcf().dpi)
    #cardiac_explainer = shap.TreeExplainer(cardiac_loaded_model, data=data_df[cardiac_top_features])
    #cardiac_shap_values = cardiac_explainer.shap_values(data_df[cardiac_top_features])
    # shap.force_plot(cardiac_explainer.expected_value[1], cardiac_shap_values[1][-50,:], feature_names=cardiac_top_features,link='identity', contribution_threshold=0.9, matplotlib=False)
    pass

def _save_tree_explainer_and_shaps():
    data_df= pd.read_csv('.\\data\\data_100.csv')
    image_list_str = []
    lim_types = ['cardiac', 'pulmonary', 'other']
    selected_model = None
    feature_selector = None
    for lim_type in lim_types:
        if lim_type == 'cardiac':
            selected_model = pickle.load(
                open('.\\models\\cardiac\\clf_cardiac_100.sav', 'rb'))
            feature_selector = cardiac_data_100[1:]
        elif lim_type == 'pulmonary':
            selected_model = pickle.load(
                open('.\\models\\pulmonary\\clf_pulmonary_100.sav', 'rb'))
            feature_selector = pulmonary_data_100[1:]
        else:
            selected_model = pickle.load(
                open('.\\models\\other\\clf_other_100.sav', 'rb'))
            feature_selector = other_data_100[1:]
        explainer = shap.TreeExplainer(selected_model, data=data_df[feature_selector])
        shap_values = explainer.shap_values(data_df[feature_selector])
        print(explainer)
        print(shap_values)
        pickle.dump(explainer, open(f".\\models\\{lim_type}\\"+lim_type+'_tree_explainer.sav','wb'))
        pickle.dump(shap_values, open(f".\\models\\{lim_type}\\"+lim_type+'_shap_values.sav','wb'))
        loaded_explainer = pickle.load(
                    open(f".\\models\\{lim_type}\\"+lim_type+'_tree_explainer.sav', 'rb'))
        loaded_shap_values = pickle.load(
                    open(f".\\models\\{lim_type}\\"+lim_type+'_shap_values.sav', 'rb'))

        print(loaded_explainer)
        print(loaded_shap_values)
        print(str(type(loaded_shap_values)))
    pass

def get_cardiac_cpet_intepretation_by_id(session_id, lim_type):
    try:
        data_df= pd.read_csv('.\\data\\data_100.csv')
        session_id = float(session_id)
        selected_model = None
        feature_selector = None
        if lim_type == 'cardiac':
            selected_model = pickle.load(
                open('.\\models\\cardiac\\clf_cardiac_100.sav', 'rb'))
            feature_selector = cardiac_data_100[1:]
        elif lim_type == 'pulmonary':
            selected_model = pickle.load(
                open('.\\models\\pulmonary\\clf_pulmonary_100.sav', 'rb'))
            feature_selector = pulmonary_data_100[1:]
        else:
            selected_model = pickle.load(
                open('.\\models\\other\\clf_other_100.sav', 'rb'))
            feature_selector = other_data_100[1:]
        explainer = shap.TreeExplainer(selected_model, data=data_df[feature_selector])
        shap_values = explainer.shap_values(data_df[feature_selector])
        data_index = data_df.loc[data_df['SessionId'] == session_id].index[0]
        pl_result = summary_with_highlight(shap_values[1], data_df[feature_selector], row_highlight=data_index, max_display=10, as_string=True)
        return pl_result, 200
    except Exception as e:
        return "Unexpected error", 400
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
    create_force_plot_string('cardiac', 7)
    _test_force_plot()
    #_save_tree_explainer_and_shaps()
    #sget_cardiac_cpet_intepretation_by_id("7", "cardiac")
    pass

