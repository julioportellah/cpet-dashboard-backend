import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import io
import matplotlib.pyplot as pl
import shap
import base64
from custom_shap import summary_with_highlight 

cardiac_data_100 = ['DiffPercentPeakVO2', 'DiffPeakVO2','75_to_100_VO2Slope','75_to_100_HRSlope','MinO2Pulse',
                      'PeakVE','VO2vsPeakVO2atVT','second_half_RRSlope','second_half_VO2Slope','75_to_100_VCO2Slope','MeanVE',
                      'second_half_VESlope','O2PulseDiff','50_to_75_O2Slope',
                        'O2PulsePercent','75_to_100_RERSlope','PeakRER','50_to_75_VO2Slope','PeakVO2Real']
cardiac_data_90 = ['DiffPercentPeakVO2','DiffPeakVO2','MinO2Pulse','second_half_VEVO2Slope',
                '25_to_50_VCO2Slope','VO2vsPeakVO2atVT','PeakVE','15_to_85_VO2Slope','first_half_VEVCO2Slope',
                '25_to_50_VO2Slope','MeanVO2','25_to_50_VESlope','PeakVO2Real','PeakVO2', '15_to_85_VESlope',
                   'second_half_RRSlope','PeakRER','second_half_VCO2Slope','O2PulsePercent','15_to_85_VCO2Slope',
                   '75_to_100_VESlope','MeanVE','first_half_VO2Slope','second_half_VESlope','first_half_VEVO2Slope',
                   '75_to_100_HRSlope','DiffPercentPeakHR','15_to_85_HRSlope']
cardiac_data_80 = ['DiffPercentPeakVO2','DiffPeakVO2','first_half_VEVCO2Slope','MinO2Pulse','15_to_85_VO2Slope',
                   '25_to_50_VEVCO2Slope','MeanVO2','PeakVO2Real','PeakVO2','MeanHeartRate','first_half_VO2Slope',
                  'first_half_VEVO2Slope','O2PulsePercent','StdHeartRate','second_half_VEVO2Slope','75_to_100_RRSlope',
                  'DiffPeakHR','PredictedMaxHR']
cardiac_data_70 = ['DiffPercentPeakVO2','15_to_85_VO2Slope','PeakVO2Real','DiffPeakVO2','MinO2Pulse','PeakVO2',
                   'first_half_VEVCO2Slope','MeanVO2','15_to_85_VCO2Slope','PeakVCO2','StdO2Pulse','PredictedMaxHR',
                   '15_to_85_VESlope','MeanVCO2']
cardiac_data_60 = ['50_to_75_VO2Slope','50_to_75_VCO2Slope','DiffPercentPeakVO2','15_to_85_VO2Slope','StdO2Pulse',
                  '50_to_75_VESlope','first_half_VEVCO2Slope','PeakVO2','MinO2Pulse','PeakVO2Real','O2PulsePercent',
                  'MeanVO2','StdHeartRate','LowestVE/VCO2','VEvsVCO2Slope','50_to_75_HRSlope',
                  '15_to_85_VEVO2Slope']
cardiac_data_50 = ['StdO2Pulse','15_to_85_VEVCO2Slope','DiffPercentPeakVO2','second_half_VCO2Slope',
                'MeanVE/VCO2','second_half_VO2Slope','second_half_VESlope','PeakVO2Real','MinO2Pulse','PeakVO2','O2PulseDiff',
                'StdHeartRate','15_to_85_VEVO2Slope','VEvsVCO2Slope','second_half_HRSlope']
cardiac_data_40 = ['StdO2Pulse','DiffPercentPeakVO2','second_half_VEVCO2Slope','MinO2Pulse','O2PulseDiff',
                   'PeakVO2','PeakVO2Real','O2PulsePercent','VEvsVCO2Slope','MaxO2Pulse','LowestVE/VCO2','MeanVE/VCO2',
                  'second_half_VEVO2Slope','second_half_HRSlope','second_half_VCO2Slope','PeakVCO2','StdVE/VCO2']


pulmonary_data_100 = ['O2PulsePercent', 'O2PulseDiff','first_half_VO2Slope','LowestVE/VCO2',
                      'first_half_VCO2Slope', '15_to_85_RRSlope','PeakRR','50_to_75_RRSlope','MeanO2Pulse','VEvsVCO2Slope',
                     '25_to_50_VCO2Slope','StdHeartRate']
pulmonary_data_90 = ['O2PulsePercent','O2PulseDiff','second_half_RRSlope','LowestVE/VCO2',
                    'second_half_VESlope','PeakRR','75_to_100_VESlope','75_to_100_RRSlope','PeakVE','VEvsVCO2Slope',
                    'MaxO2Pulse','StdHeartRate','MeanVO2','MeanVE']
pulmonary_data_80 = ['O2PulsePercent','O2PulseDiff','LowestVE/VCO2','first_half_VO2Slope',
                    'first_half_VCO2Slope','VEvsVCO2Slope','75_to_100_RRSlope','MeanO2Pulse']
pulmonary_data_70 = ['O2PulsePercent','O2PulseDiff','LowestVE/VCO2','VEvsVCO2Slope','MaxO2Pulse',
                     'second_half_RRSlope','first_half_VO2Slope','MeanVCO2','MeanVO2']
pulmonary_data_60 = ['O2PulsePercent','O2PulseDiff','LowestVE/VCO2','MeanVO2','MeanVCO2','second_half_RRSlope',
                    'MaxO2Pulse','15_to_85_VCO2Slope','second_half_VEVO2Slope','PeakVO2Real']
pulmonary_data_50 = ['O2PulsePercent','O2PulseDiff','DiffPercentPeakVO2','15_to_85_VO2Slope','75_to_100_RRSlope',
                     'PeakVE','MeanVO2','MaxO2Pulse','MeanVCO2','PeakVO2','PeakVCO2','MaxO2_EST']
pulmonary_data_40 = ['O2PulsePercent','O2PulseDiff','DiffPercentPeakVO2','PeakVE','PeakVO2',
                    'PeakVO2Real','MaxO2Pulse','StdO2Pulse','MeanRR','PeakVCO2','MeanVO2','MeanVE',
                    'DiffPercentPeakHR','MeanHeartRate','LowestRR','DiffPeakHR','StdHeartRate','LowestVE/VCO2']

other_data_100 = ['PeakRR', 'PeakVE','PeakVCO2','MeanVCO2','PeakVO2','PeakVO2Real',
                  'LowestVE/VCO2','MeanRER','PeakRER','VO2vsPeakVO2atVT','DiffPercentPeakVO2','MeanRR',
                  '75_to_100_VEVCO2Slope','DiffPeakVO2','MeanVE','second_half_VESlope','first_half_VEVCO2Slope',
                  '0_to_25_O2Slope','VO2atVT', 'MeanVO2','second_half_VCO2Slope','DiffPeakHR','MeanVE/VCO2','75_to_100_RRSlope']
other_data_90 = ['PeakVE','PeakRR','PeakVCO2','DiffPercentPeakVO2','PeakVO2','PeakVO2Real',
                'MeanRER','MeanRR','0_to_25_O2Slope','DiffPeakVO2','LowestVE/VCO2','MeanVCO2','MeanO2Pulse',
                 '75_to_100_HRSlope','MeanVE','0_to_25_VESlope','MeanVO2','VEvsVCO2Slope','0_to_25_VO2Slope',
                'second_half_VEVCO2Slope']
other_data_80 = ['PeakVCO2','PeakRR','PeakVE','LowestVE/VCO2','DiffPercentPeakVO2','MeanVCO2',
                'PeakVO2','PeakVO2Real','O2PulseDiff','DiffPeakVO2','MeanVO2','O2PulsePercent','0_to_25_HRSlope',
                 'MeanVE','MeanO2Pulse','VEvsVCO2Slope','MeanRER','PeakRER','second_half_VCO2Slope','15_to_85_RRSlope',
                 '50_to_75_VEVCO2Slope']
other_data_70 = ['first_half_O2Slope','PeakVCO2','DiffPercentPeakVO2','LowestVE/VCO2','PeakVE',
                'PeakVO2','PeakVO2Real','PeakRR','O2PulseDiff','O2PulsePercent','MeanVCO2','DiffPeakVO2',
                '0_to_25_VO2Slope','MeanO2Pulse','50_to_75_VO2Slope','MeanVO2','0_to_25_O2Slope','DiffPercentPeakHR']
other_data_60 = ['first_half_O2Slope','LowestVE/VCO2','PeakVCO2','PeakVE','DiffPercentPeakVO2',
                'O2PulsePercent','PeakVO2','PeakVO2Real','15_to_85_VEVCO2Slope','O2PulseDiff',
                 '0_to_25_HRSlope','MeanVCO2','first_half_VO2Slope','MeanVO2','PeakRR','MeanO2Pulse','second_half_VO2Slope',
                'MeanVE/VCO2','VEvsVCO2Slope','0_to_25_VCO2Slope']
other_data_50 = ['first_half_O2Slope','LowestVE/VCO2','25_to_50_O2Slope','0_to_25_HRSlope',
                 'first_half_VESlope','75_to_100_VO2Slope','first_half_VO2Slope','PeakVCO2','MeanVCO2','MeanVE/VCO2',
                '75_to_100_O2Slope','75_to_100_VCO2Slope','VEvsVCO2Slope','DiffPercentPeakVO2','HRvsVO2Slope',
                'PeakVO2','50_to_75_VEVCO2Slope']
other_data_40 = ['LowestVE/VCO2','MeanVCO2','first_half_HRSlope','MeanVO2','first_half_VO2Slope',
                'PeakVCO2','O2PulseDiff','MeanVE/VCO2','O2PulsePercent','PeakVO2','first_half_O2Slope',
                'second_half_VCO2Slope','PeakVE','MeanO2Pulse']


class NewPatientDynamicFullPrediction():
    def __init__(self, cardiac_proba_array, pulmonary_proba_array,
                 other_proba_array,cardiac_force=None,pulmonary_force=None,
                 other_force=None, cardiac_summary=None, pulmonary_summary=None,
                 other_summary=None) -> None:
        self.cardiac_proba = cardiac_proba_array
        self.pulmonary_proba = pulmonary_proba_array
        self.other_proba = other_proba_array
        self.time_list = [40, 50, 60, 70, 80, 90, 100]
        self.cardiac_force = cardiac_force
        self.pulmonary_force = pulmonary_force
        self.other_force = other_force
        pass
    pass

def process_data(df):
    df.columns = df.columns.str.replace(' ', '')
    df['PatientId']=-1
    df['SessionId']=-1
    df['BMI'] = df['weight-kg']/(df['height-cm']/100)**2
    df['MaxVO2_EST']=df.apply(lambda x: get_vo2_peak(x),axis=1)
    df['MaxO2_EST']=df.apply(lambda x: get_o2_peak(x),axis=1)
    df['PredictedMaxHR']=df.apply(lambda x: max_hr_predicted(x),axis=1)
    df.TIME = df.TIME.str.replace(' ','')
    minutes = []
    for i in df.TIME:
        minute, second = i.split(":")
        minutes.append(float(second)/60+float(minute))
    df['minutes'] = np.array(minutes)
    #Getting patient's max time
    patient_times = df.groupby(['PatientId','SessionId'])['minutes'].max().reset_index()
    patient_times['max_time'] = patient_times.apply(round_to_30s, axis=1)
    #Generating the binned dataframe
    df_info_avg = pd.DataFrame(columns = ['PatientId','SessionId','minutes'])
    for i in patient_times.SessionId:
        patient_time_min = np.arange(0.5,patient_times.loc[(patient_times.PatientId==np.floor(i)) & 
                                                        (patient_times.SessionId==i)]['max_time'].values[0],.5)
        patient_ids=np.ones(patient_time_min.shape)*np.floor(i)
        patient_session_ids = np.ones(patient_time_min.shape)*i
        tempdf = pd.DataFrame({'PatientId': patient_ids, 'SessionId':patient_session_ids, 'minutes': patient_time_min })
        df_info_avg=df_info_avg.append(tempdf, ignore_index=True)
        pass

    #Adding features
    df_info_avg['HR']=np.round(df_info_avg.apply(lambda x: round_to_mean('HR',x,df),axis=1),0)
    df_info_avg['VO2']=df_info_avg.apply(lambda x: round_to_mean('VO2',x,df),axis=1)
    df_info_avg['VCO2']=df_info_avg.apply(lambda x: round_to_mean('VCO2',x,df),axis=1)
    df_info_avg['VE']=df_info_avg.apply(lambda x: round_to_mean('VE',x,df),axis=1)
    df_info_avg['VE/VO2']=df_info_avg.apply(lambda x: round_to_mean('VE/VO2',x,df),axis=1)
    df_info_avg['VE/VCO2']=df_info_avg.apply(lambda x: round_to_mean('VE/VCO2',x,df),axis=1)
    df_info_avg['RER']=df_info_avg.apply(lambda x: round_to_mean('RER',x,df),axis=1)
    df_info_avg['RR']=df_info_avg.apply(lambda x: round_to_mean('RR',x,df),axis=1)
    df_info_avg['HasAnaerobicThreshold']=df_info_avg.apply(lambda x: has_anaerobic_threshold(x,df),axis=1)
    df_info_avg['sex']=df['sex'].values[0]
    df_info_avg['age']=df['age'].values[0]
    df_info_avg['BMI']=df['BMI'].values[0]
    df_info_avg['MaxVO2_EST']=df['MaxVO2_EST'].values[0]
    df_info_avg['MaxO2_EST']=df['MaxO2_EST'].values[0]
    df_info_avg['PredictedMaxHR']=df['PredictedMaxHR'].values[0]
    df_info_avg['O2']=df_info_avg.VO2/df_info_avg.HR*1000
    df_info_avg['maxLocalTime'] = df_info_avg.groupby('SessionId')['minutes'].transform('max')


    data_40 = generate_cpet_data_by_time_percentage(df_info_avg, df, 'file_name', 40)
    data_50 = generate_cpet_data_by_time_percentage(df_info_avg, df, 'file_name', 50)
    data_60 = generate_cpet_data_by_time_percentage(df_info_avg, df, 'file_name', 60)
    data_70 = generate_cpet_data_by_time_percentage(df_info_avg, df, 'file_name', 70)
    data_80 = generate_cpet_data_by_time_percentage(df_info_avg, df, 'file_name', 80)
    data_90 = generate_cpet_data_by_time_percentage(df_info_avg, df, 'file_name', 90)
    data_100 = generate_cpet_data_by_time_percentage(df_info_avg, df, 'file_name', 100)
    result_datasets = [data_40, data_50, data_60, data_70, data_80, data_90, data_100]

    cardiac_col_list = [cardiac_data_40, cardiac_data_50, cardiac_data_60, cardiac_data_70, cardiac_data_80, cardiac_data_90,
            cardiac_data_100]
    cardiac_scaler_list = _generate_list_loaded_scaler('cardiac')
    cardiac_model_list = _generate_list_loaded_models('cardiac')
    cardiac_dynamic_result = _generate_list_predictions(result_datasets, cardiac_col_list, cardiac_scaler_list, cardiac_model_list)
    
    pulmonary_col_list = [pulmonary_data_40, pulmonary_data_50, pulmonary_data_60, pulmonary_data_70, pulmonary_data_80, 
                        pulmonary_data_90, pulmonary_data_100]
    pulmonary_scaler_list = _generate_list_loaded_scaler('pulmonary')
    pulmonary_model_list = _generate_list_loaded_models('pulmonary')
    pulmonary_dynamic_result = _generate_list_predictions(result_datasets, pulmonary_col_list, pulmonary_scaler_list, pulmonary_model_list)
    
    other_col_list = [other_data_40, other_data_50, other_data_60, other_data_70, other_data_80, other_data_90, other_data_100]
    other_scaler_list = _generate_list_loaded_scaler('other')
    other_model_list = _generate_list_loaded_models('other')
    other_dynamic_result = _generate_list_predictions(result_datasets, other_col_list, other_scaler_list, other_model_list)
    time_list = [40, 50, 60, 70, 80, 90, 100]
    #Adding the force plot
    cardiac_force_plot = create_force_plot_string('cardiac',data_100,cardiac_col_list[-1])
    pulmonary_force_plot = create_force_plot_string('pulmonary',data_100,pulmonary_col_list[-1])
    other_force_plot = create_force_plot_string('other',data_100,other_col_list[-1])
    
    result = NewPatientDynamicFullPrediction(cardiac_dynamic_result, pulmonary_dynamic_result, other_dynamic_result,
                                            cardiac_force_plot[0], pulmonary_force_plot[0], other_force_plot[0],
                                            cardiac_force_plot[1], pulmonary_force_plot[1], other_force_plot[1])
    # lim_type = 'cardiac'
    # loaded_tree = pickle.load(open(f".\\models\\{lim_type}\\"+lim_type+'_tree_explainer.sav', 'rb'))
    # shap_values = loaded_tree.shap_values(data_100[cardiac_col_list[-1]])
    # print(shap_values[1])
    return result
    pass

def create_force_plot_string(lim_type, df, cols):
    loaded_tree = pickle.load(open(f".\\models\\{lim_type}\\"+lim_type+'_tree_explainer.sav', 'rb'))
    loaded_previous_shaps = pickle.load(open(f".\\models\\{lim_type}\\"+lim_type+'_shap_values.sav', 'rb'))
    shap_values = loaded_tree.shap_values(df[cols])
    loaded_df = pd.read_csv('.\\data\\data_100.csv')
    loaded_df = loaded_df[cols]

    my_stringIObytes = io.BytesIO()
    shap.force_plot(loaded_tree.expected_value[1], shap_values[1][0], feature_names=cols,
                    link='identity', contribution_threshold=0.1, show=False, plot_cmap=['#77dd77', '#f99191'],
                    matplotlib=True).savefig(my_stringIObytes, format="png", dpi=150, bbox_inches='tight')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.getvalue()).decode("utf-8").replace("\n", "")
    pl.close()
    ##
    all_shap_values = np.append(loaded_previous_shaps[1], shap_values[1], axis=0)
    loaded_df = loaded_df.append(df[cols])
    pl_result = summary_with_highlight(all_shap_values, loaded_df[cols], row_highlight=-1, max_display=10, as_string=True)
    pl.close()
    return str(my_base64_jpgData), str(pl_result)


def _generate_list_predictions(list_dfs, list_cols, list_scalers, list_models):
    result = []
    for i in range(len(list_dfs)):
        data_filtered = list_dfs[i][list_cols[i]]
        X_scaled = list_scalers[i].transform(data_filtered)
        y_hat = np.round(list_models[i].predict_proba(X_scaled),2)*100
        
        result.append(int(y_hat[:,1][0]))
        pass
    return result

def _generate_list_loaded_models(type):
    result = []
    time_list = [40, 50, 60, 70, 80, 90, 100]
    for time in time_list:
        selected_model = pickle.load(
                open(f'.\\models\\{type}\\clf_{type}_{time}.sav', 'rb'))
        result.append(selected_model)
        pass
    return result

def _generate_list_loaded_scaler(type):
    result = []
    time_list = [40, 50, 60, 70, 80, 90, 100]
    for time in time_list:
        selected_model = pickle.load(
                open(f'.\\models\\{type}\\scaler_clf_{type}_{time}.sav', 'rb'))
        result.append(selected_model)
        pass
    return result

#Functions
# BASIC FUNCTIONS
def get_lowest_variable(column, row, df):
    return np.min(df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)][column].values)

def get_highest_variable(column, row, df):
    return np.max(df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)][column].values)

def get_mean_variable(column, row, df):
    return np.mean(df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)][column].values)

def get_std_variable(column, row, df):
    return np.std(df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)][column].values)

def get_median_variable(column, row, df):
    return np.median(df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)][column].values)

#HR FUNCTIONS
def get_HR_percent(row,df):
    max_hr_real = np.max(df.loc[(df.PatientId == row.PatientId) & (df.SessionId == row.SessionId)]['HR'].values)
    max_hr_expected = row.PredictedMaxHR
    return max_hr_real/max_hr_expected
def get_HR_diff(row, df):
    max_hr_real = np.max(df.loc[(df.PatientId == row.PatientId) & (df.SessionId == row.SessionId)]['HR'].values)
    max_hr_expected = row.PredictedMaxHR
    return max_hr_real-max_hr_expected
#VO2 FUNCTIONS
#VO2 vs expected VO2 functions
def get_VO2_percent(row,df):
    max_vo2_real = np.max(df.loc[(df.PatientId == row.PatientId) & (df.SessionId == row.SessionId)]['VO2'].values)
    max_vo2_expected = row.MaxVO2_EST
    return max_vo2_real/max_vo2_expected

def get_MaxVO2_expected_vs_real(row, df):
    max_vo2_real = np.max(df.loc[(df.PatientId == row.PatientId) & (df.SessionId == row.SessionId)]['VO2'].values)
    max_vo2_expected = row.MaxVO2_EST
    return max_vo2_real-max_vo2_expected

#AGAINST VO2 SLOPE FUNCTIONS
# Slope against VO2 functions
def get_HR_VO2_slope(row,df):
    lin_reg = LinearRegression()
    y_val = df.loc[(df.PatientId == row.PatientId) & (df.SessionId == row.SessionId)]['HR'].values
    X_val = df.loc[(df.PatientId == row.PatientId) & (df.SessionId == row.SessionId)]['VO2'].values.reshape(-1, 1)
    lin_reg.fit(X_val,y_val)
    return lin_reg.coef_[0]

def get_VE_VCO2_slope(row,df):
    lin_reg = LinearRegression()
    y_val = df.loc[(df.PatientId == row.PatientId) & (df.SessionId == row.SessionId)]['VE'].values
    X_val = df.loc[(df.PatientId == row.PatientId) & (df.SessionId == row.SessionId)]['VCO2'].values.reshape(-1, 1)
    lin_reg.fit(X_val,y_val)
    return lin_reg.coef_[0]

# BASIC O2 PULSE FUNCTIONS
def get_mean_O2_pulse(row, df):
    hr_values = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['HR'].values
    zero_ind = np.where(hr_values == 0)[0]
    if len(hr_values) == len(zero_ind):
        return -1
    vo2_values =df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['VO2'].values
    hr_values = np.delete(hr_values, zero_ind)
    vo2_values = np.delete(vo2_values, zero_ind)
    O2_pulse = vo2_values/hr_values
    return np.mean(O2_pulse)

def get_max_O2_pulse(row, df):
    hr_values = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['HR'].values
    zero_ind = np.where(hr_values == 0)[0]
    if len(hr_values) == len(zero_ind):
        return -1
    vo2_values =df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['VO2'].values
    hr_values = np.delete(hr_values, zero_ind)
    vo2_values = np.delete(vo2_values, zero_ind)
    O2_pulse = vo2_values/hr_values
    return np.max(O2_pulse)

def get_min_O2_pulse(row, df):
    hr_values = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['HR'].values
    zero_ind = np.where(hr_values == 0)[0]
    if len(hr_values) == len(zero_ind):
        return -1
    vo2_values =df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['VO2'].values
    hr_values = np.delete(hr_values, zero_ind)
    vo2_values = np.delete(vo2_values, zero_ind)
    O2_pulse = vo2_values/hr_values
    return np.min(O2_pulse)

def get_std_O2_pulse(row, df):
    hr_values = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['HR'].values
    zero_ind = np.where(hr_values == 0)[0]
    if len(hr_values) == len(zero_ind):
        return -1
    vo2_values =df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['VO2'].values
    hr_values = np.delete(hr_values, zero_ind)
    vo2_values = np.delete(vo2_values, zero_ind)
    O2_pulse = vo2_values/hr_values
    return np.std(O2_pulse)

def get_median_O2_pulse(row, df):
    hr_values = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['HR'].values
    zero_ind = np.where(hr_values == 0)[0]
    if len(hr_values) == len(zero_ind):
        return -1
    vo2_values =df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['VO2'].values
    hr_values = np.delete(hr_values, zero_ind)
    vo2_values = np.delete(vo2_values, zero_ind)
    O2_pulse = vo2_values/hr_values
    return np.median(O2_pulse)

#O2 BENCHMARK FUNCTIONS
def get_O2_percent(row,df):
    max_o2_real = np.max(df.loc[(df.PatientId == row.PatientId) & (df.SessionId == row.SessionId)]['O2'].values)
    max_o2_expected = row.MaxO2_EST
    return max_o2_real/max_o2_expected

def get_O2_diff(row,df):
    max_o2_real = np.max(df.loc[(df.PatientId == row.PatientId) & (df.SessionId == row.SessionId)]['O2'].values)
    max_o2_expected = row.MaxO2_EST
    return max_o2_real-max_o2_expected


# SLOPE FUNCTIONS
def get_first_quarter(row, df, column):
    time_span=df.loc[df.SessionId == row.SessionId].minutes.values.max()-df.loc[df.SessionId == row.SessionId].minutes.values.min()
    last_quarter_time = df.loc[df.SessionId == row.SessionId].minutes.values.min()+.25*time_span
    initial_quarter_time = df.loc[df.SessionId == row.SessionId].minutes.values.min()
    y_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_quarter_time)& 
                   (df.minutes <= last_quarter_time)][column].values
    X_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_quarter_time)& 
                   (df.minutes <= last_quarter_time)].minutes.values
    lin_reg = LinearRegression()
    lin_reg.fit(X_val.reshape(-1, 1),y_val.reshape(-1, 1))
    return lin_reg.coef_[0][0]

def get_second_quarter(row, df, column):
    time_span=df.loc[df.SessionId == row.SessionId].minutes.values.max()-df.loc[df.SessionId == row.SessionId].minutes.values.min()
    last_quarter_time = df.loc[df.SessionId == row.SessionId].minutes.values.min()+.5*time_span
    initial_quarter_time = df.loc[df.SessionId == row.SessionId].minutes.values.min()+.25*time_span
    y_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_quarter_time)& 
                   (df.minutes <= last_quarter_time)][column].values
    X_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_quarter_time)& 
                   (df.minutes <= last_quarter_time)].minutes.values
    lin_reg = LinearRegression()
    lin_reg.fit(X_val.reshape(-1, 1),y_val.reshape(-1, 1))
    return lin_reg.coef_[0][0]

def get_third_quarter(row, df, column):
    time_span=df.loc[df.SessionId == row.SessionId].minutes.values.max()-df.loc[df.SessionId == row.SessionId].minutes.values.min()
    last_quarter_time = df.loc[df.SessionId == row.SessionId].minutes.values.min()+.75*time_span
    initial_quarter_time = df.loc[df.SessionId == row.SessionId].minutes.values.min()+.5*time_span
    y_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_quarter_time)& 
                   (df.minutes <= last_quarter_time)][column].values
    X_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_quarter_time)& 
                   (df.minutes <= last_quarter_time)].minutes.values
    lin_reg = LinearRegression()
    lin_reg.fit(X_val.reshape(-1, 1),y_val.reshape(-1, 1))
    return lin_reg.coef_[0][0]

def get_last_quarter(row, df, column):
    time_span=df.loc[df.SessionId == row.SessionId].minutes.values.max()-df.loc[df.SessionId == row.SessionId].minutes.values.min()
    last_quarter_time = df.loc[df.SessionId == row.SessionId].minutes.values.min()+time_span
    initial_quarter_time = df.loc[df.SessionId == row.SessionId].minutes.values.min()+.75*time_span
    y_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_quarter_time)& 
                   (df.minutes <= last_quarter_time)][column].values
    X_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_quarter_time)& 
                   (df.minutes <= last_quarter_time)].minutes.values
    lin_reg = LinearRegression()
    lin_reg.fit(X_val.reshape(-1, 1),y_val.reshape(-1, 1))
    return lin_reg.coef_[0][0]

def get_first_half(row, df, column):
    time_span=df.loc[df.SessionId == row.SessionId].minutes.values.max()-df.loc[df.SessionId == row.SessionId].minutes.values.min()
    last_half_time = df.loc[df.SessionId == row.SessionId].minutes.values.min()+.5*time_span
    initial_half_time = df.loc[df.SessionId == row.SessionId].minutes.values.min()
    y_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_half_time)& 
                   (df.minutes <= last_half_time)][column].values
    X_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_half_time)& 
                   (df.minutes <= last_half_time)].minutes.values
    lin_reg = LinearRegression()
    lin_reg.fit(X_val.reshape(-1, 1),y_val.reshape(-1, 1))
    return lin_reg.coef_[0][0]

def get_second_half(row, df, column):
    time_span=df.loc[df.SessionId == row.SessionId].minutes.values.max()-df.loc[df.SessionId == row.SessionId].minutes.values.min()
    last_half_time = df.loc[df.SessionId == row.SessionId].minutes.values.min()+time_span
    initial_half_time = df.loc[df.SessionId == row.SessionId].minutes.values.min()+.5*time_span
    y_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_half_time)& 
                   (df.minutes <= last_half_time)][column].values
    X_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_half_time)& 
                   (df.minutes <= last_half_time)].minutes.values
    lin_reg = LinearRegression()
    lin_reg.fit(X_val.reshape(-1, 1),y_val.reshape(-1, 1))
    return lin_reg.coef_[0][0]

def get_slope_15_85(row, df, column):
    last_quarter_time = df.loc[df.SessionId == row.SessionId].minutes.values.max()*.85
    initial_quarter_time = df.loc[df.SessionId == row.SessionId].minutes.values.max()*.15
    y_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_quarter_time)& 
                   (df.minutes <= last_quarter_time)][column].values
    X_val = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) & 
                   (df.minutes >= initial_quarter_time)& 
                   (df.minutes <= last_quarter_time)].minutes.values
    lin_reg = LinearRegression()
    lin_reg.fit(X_val.reshape(-1, 1),y_val.reshape(-1, 1))
    return lin_reg.coef_[0][0]

#AT FUNCTIONS
def get_vt_time(row, df):
    df_vals = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)][['VE/VCO2','VE/VO2','minutes']]
    minmax = np.max(df_vals.minutes.values)
    df_selected = df_vals.loc[(df_vals['VE/VO2']>=df_vals['VE/VCO2']) & (df_vals.minutes>=(minmax*1/2.5))]
    if df_selected.shape[0]<=2:
        return -1
    for minute in df_selected.minutes.values:
        df_vals_times = df_vals.loc[(df_vals.minutes>=minute-0.5) & (df_vals.minutes<=minute+1)]
        if df_vals_times.shape[0]<4:
            continue
        is_minor = np.round(df_vals_times.iloc[0]['VE/VCO2'])>= np.round(df_vals_times.iloc[0]['VE/VO2'])
        is_growing_1 = df_vals_times.iloc[1]['VE/VO2']< df_vals_times.iloc[2]['VE/VO2']
        is_growing_2 = df_vals_times.iloc[1]['VE/VO2']< df_vals_times.iloc[3]['VE/VO2']
        if((is_minor & is_growing_1) | (is_minor & is_growing_2))==True:
            return minute
        pass
    return -1
    pass

def get_vt_index(row, df):
    df_vals = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)][['VE/VCO2','VE/VO2','minutes']]
    minutes_list = df_vals.minutes
    minmax = np.max(df_vals.minutes.values)
    df_selected = df_vals.loc[(df_vals['VE/VO2']>=df_vals['VE/VCO2']) & (df_vals.minutes>=(minmax*1/2.5))]
    if df_selected.shape[0]<=2:
        return -1
    for minute in df_selected.minutes.values:
        df_vals_times = df_vals.loc[(df_vals.minutes>=minute-0.5) & (df_vals.minutes<=minute+1)]
        if df_vals_times.shape[0]<4:
            continue
        is_minor = np.round(df_vals_times.iloc[0]['VE/VCO2'])>= np.round(df_vals_times.iloc[0]['VE/VO2'])
        is_growing_1 = df_vals_times.iloc[1]['VE/VO2']< df_vals_times.iloc[2]['VE/VO2']
        is_growing_2 = df_vals_times.iloc[1]['VE/VO2']< df_vals_times.iloc[3]['VE/VO2']
        if((is_minor & is_growing_1) | (is_minor & is_growing_2))==True:
            index = np.where(minutes_list == minute)
            return index[0][0]
        pass
    return -1
    pass

def get_VO2atVT(row, df):
    rer_list = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['RER'].values
    at_index = get_vt_index(row, df)
    if at_index == -1:
        return -1
    vo2_list = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['VO2'].values
    return vo2_list[at_index]

def get_percent_time_after_VT(row, df):
    rer_list = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['RER'].values
    at_index = get_vt_index(row, df)
    if at_index == -1:
        return 0
    df_times = df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['minutes'].values
    return df_times[at_index]/df_times[-1]

# Round the values within the 30s window
def round_to_mean(column, row, df):
    return np.mean(df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId) &
                   (df.minutes > (row.minutes-0.5)) &
                   (df.minutes <= (row.minutes))][column].values)
# Function that does the left join without shenaningans
def place_label_fake_join(column, row, df):
    return df.loc[(df.PatientId == row.PatientId)][column].values[0]
# Superficial function to detect anaerobic threshold
def has_anaerobic_threshold(row, df):
    return np.max(df.loc[(df.PatientId == row.PatientId) &
                   (df.SessionId == row.SessionId)]['RER'].values) >= 1



def round_to_30s(row):
    base = np.floor(row.minutes)
    med = base +0.5
    sup = np.round(row.minutes)
    if row.minutes <= med:
        return med
    else:
        return sup
    pass

def get_vo2_peak(row):
    sex = 0
    bmi = 0
    age = row.age
    if row.sex == 'F':
        sex=2
    else:
        sex=1
    if row.BMI <= 25:
        bmi = 0
    else:
        bmi = 1
    if age < 34:
        age = 1
    elif 34 <= age < 44:
        age = 2
    elif 44 <= age < 54:
        age = 3
    elif 54 <= age < 64:
        age = 4
    elif age >=64:
        age = 5
    peak_est = 47.7565 -0.9880*age-0.2356*age**2-8.8697*sex+2.3597*bmi-2.0308*age*bmi-3.7405*sex*bmi+0.2512*age*sex+1.3797*age*sex*bmi
    peak_est = peak_est * 10**-3 * row['weight-kg']
    return peak_est

def get_o2_peak(row):
    sex = 0
    bmi = 0
    age = row.age
    if row.sex == 'F':
        sex=2
    else:
        sex=1
    if row.BMI <= 25:
        bmi = 0
    else:
        bmi = 1
    if age < 34:
        age = 1
    elif 34 <= age < 44:
        age = 2
    elif 44 <= age < 54:
        age = 3
    elif 54 <= age < 64:
        age = 4
    elif age >=64:
        age = 5
    peak_est = 22.1667 -0.8167*age+0.0167*age**2-5.8667*sex+4.8897*bmi-1.4230*age*bmi-2.4230*sex*bmi+0.3333*age*sex+0.7897*age*sex*bmi
    peak_est = peak_est * 10**-3 
    return peak_est

def max_hr_predicted(row):
    peak_hr = 208 - 0.7*row['age']
    return peak_hr


def generate_cpet_data_by_time_percentage(df_data_sampled, df_patient_info,  file_name, time_percent=100):
    #getting sessions
    data_resume_cpet_df = pd.DataFrame({ 'SessionId':df_data_sampled.SessionId.drop_duplicates().values})
    data_resume_cpet_df['PatientId'] = np.floor(data_resume_cpet_df.SessionId)
    df_data_sampled = df_data_sampled.dropna()
    df_data_sampled = df_data_sampled.loc[df_data_sampled.minutes<=df_data_sampled.maxLocalTime*time_percent/100]
    
    
    #Joining with personal data
    data_resume_cpet_df['sex']=data_resume_cpet_df.apply(lambda x: place_label_fake_join('sex',x,df_patient_info),axis=1)
    data_resume_cpet_df['age']=data_resume_cpet_df.apply(lambda x: place_label_fake_join('age',x,df_patient_info),axis=1)
    data_resume_cpet_df['BMI']=data_resume_cpet_df.apply(lambda x: place_label_fake_join('BMI',x,df_patient_info),axis=1)
    data_resume_cpet_df['MaxVO2_EST']=data_resume_cpet_df.apply(lambda x: place_label_fake_join('MaxVO2_EST',x,df_patient_info),axis=1)
    data_resume_cpet_df['MaxO2_EST']=data_resume_cpet_df.apply(lambda x: place_label_fake_join('MaxO2_EST',x,df_patient_info),axis=1)
    data_resume_cpet_df['PredictedMaxHR']=data_resume_cpet_df.apply(lambda x: place_label_fake_join('PredictedMaxHR',x,df_patient_info),axis=1)
    # data_resume_cpet_df['CardiacLim']=data_resume_cpet_df.apply(lambda x: place_label_fake_join('BA-PrimaryCardiacLim',x,df_patient_info),axis=1)
    # data_resume_cpet_df['PulmonaryLim']=data_resume_cpet_df.apply(lambda x: place_label_fake_join('BA-PrimaryPulmonaryLim',x,df_patient_info),axis=1)
    # data_resume_cpet_df['MuscleSkeletalLim']=data_resume_cpet_df.apply(lambda x: place_label_fake_join('OtherPrimaryLim',x,df_patient_info),axis=1)
    # data_resume_cpet_df['Healthy']=data_resume_cpet_df.apply(lambda x: place_label_fake_join('Healthy',x,df_patient_info),axis=1)
    
    data_resume_cpet_df['PeakHeartRate']=data_resume_cpet_df.apply(lambda x: get_highest_variable('HR',x,df_data_sampled),axis=1)
    data_resume_cpet_df['MeanHeartRate']=data_resume_cpet_df.apply(lambda x: get_mean_variable('HR',x,df_data_sampled),axis=1)
    data_resume_cpet_df['MinHeartRate']=data_resume_cpet_df.apply(lambda x: get_lowest_variable('HR',x,df_data_sampled),axis=1)
    data_resume_cpet_df['StdHeartRate']=data_resume_cpet_df.apply(lambda x: get_std_variable('HR',x,df_data_sampled),axis=1)
    data_resume_cpet_df['LowestVE/VCO2']=data_resume_cpet_df.apply(lambda x: get_lowest_variable('VE/VCO2',x,df_data_sampled),axis=1)
    data_resume_cpet_df['PeakVE/VCO2']=data_resume_cpet_df.apply(lambda x: get_highest_variable('VE/VCO2',x,df_data_sampled),axis=1)
    data_resume_cpet_df['MeanVE/VCO2']=data_resume_cpet_df.apply(lambda x: get_mean_variable('VE/VCO2',x,df_data_sampled),axis=1)
    data_resume_cpet_df['StdVE/VCO2']=data_resume_cpet_df.apply(lambda x: get_std_variable('VE/VCO2',x,df_data_sampled),axis=1)
    data_resume_cpet_df['PeakVO2Real']=data_resume_cpet_df.apply(lambda x: get_highest_variable('VO2',x,df_data_sampled),axis=1)
    data_resume_cpet_df['DiffPeakVO2']=data_resume_cpet_df.apply(lambda x: get_MaxVO2_expected_vs_real(x,df_data_sampled),axis=1)
    data_resume_cpet_df['DiffPeakHR']=data_resume_cpet_df.apply(lambda x: get_HR_diff(x,df_data_sampled),axis=1)
    data_resume_cpet_df['DiffPercentPeakVO2']=data_resume_cpet_df.apply(lambda x: get_VO2_percent(x,df_data_sampled),axis=1)
    data_resume_cpet_df['DiffPercentPeakHR']=data_resume_cpet_df.apply(lambda x: get_HR_percent(x,df_data_sampled),axis=1)
    data_resume_cpet_df['MeanRER']=data_resume_cpet_df.apply(lambda x: get_mean_variable('RER',x,df_data_sampled),axis=1)
    data_resume_cpet_df['PeakRER']=data_resume_cpet_df.apply(lambda x: get_highest_variable('RER',x,df_data_sampled),axis=1)
    data_resume_cpet_df['LowestRER']=data_resume_cpet_df.apply(lambda x: get_lowest_variable('RER',x,df_data_sampled),axis=1)
    data_resume_cpet_df['MeanVE']=data_resume_cpet_df.apply(lambda x: get_mean_variable('VE',x,df_data_sampled),axis=1)
    data_resume_cpet_df['PeakVE']=data_resume_cpet_df.apply(lambda x: get_highest_variable('VE',x,df_data_sampled),axis=1)
    data_resume_cpet_df['LowestVE']=data_resume_cpet_df.apply(lambda x: get_lowest_variable('VE',x,df_data_sampled),axis=1)
    data_resume_cpet_df['MeanRR']=data_resume_cpet_df.apply(lambda x: get_mean_variable('RR',x,df_data_sampled),axis=1)
    data_resume_cpet_df['PeakRR']=data_resume_cpet_df.apply(lambda x: get_highest_variable('RR',x,df_data_sampled),axis=1)
    data_resume_cpet_df['LowestRR']=data_resume_cpet_df.apply(lambda x: get_lowest_variable('RR',x,df_data_sampled),axis=1)
    data_resume_cpet_df['MeanVO2']=data_resume_cpet_df.apply(lambda x: get_mean_variable('VO2',x,df_data_sampled),axis=1)
    data_resume_cpet_df['PeakVO2']=data_resume_cpet_df.apply(lambda x: get_highest_variable('VO2',x,df_data_sampled),axis=1)
    data_resume_cpet_df['LowestVO2']=data_resume_cpet_df.apply(lambda x: get_lowest_variable('VO2',x,df_data_sampled),axis=1)
    data_resume_cpet_df['MeanVCO2']=data_resume_cpet_df.apply(lambda x: get_mean_variable('VCO2',x,df_data_sampled),axis=1)
    data_resume_cpet_df['PeakVCO2']=data_resume_cpet_df.apply(lambda x: get_highest_variable('VCO2',x,df_data_sampled),axis=1)
    data_resume_cpet_df['LowestVCO2']=data_resume_cpet_df.apply(lambda x: get_lowest_variable('VCO2',x,df_data_sampled),axis=1)
    data_resume_cpet_df['HRvsVO2Slope']=data_resume_cpet_df.apply(lambda x: get_HR_VO2_slope(x,df_data_sampled),axis=1)
    data_resume_cpet_df['VEvsVCO2Slope']=data_resume_cpet_df.apply(lambda x: get_VE_VCO2_slope(x,df_data_sampled),axis=1)
    data_resume_cpet_df['MeanO2Pulse']=data_resume_cpet_df.apply(lambda x: get_mean_O2_pulse(x,df_data_sampled),axis=1)
    data_resume_cpet_df['MaxO2Pulse']=data_resume_cpet_df.apply(lambda x: get_max_O2_pulse(x,df_data_sampled),axis=1)
    data_resume_cpet_df['MinO2Pulse']=data_resume_cpet_df.apply(lambda x: get_min_O2_pulse(x,df_data_sampled),axis=1)
    data_resume_cpet_df['StdO2Pulse']=data_resume_cpet_df.apply(lambda x: get_std_O2_pulse(x,df_data_sampled),axis=1)
    data_resume_cpet_df['O2PulseDiff']=data_resume_cpet_df.MaxO2Pulse-data_resume_cpet_df.MaxO2_EST
    data_resume_cpet_df['O2PulsePercent']=data_resume_cpet_df.MaxO2Pulse/data_resume_cpet_df.MaxO2_EST
    # Given that this is an early detection case
    # First half
    data_resume_cpet_df['first_half_VO2Slope']=data_resume_cpet_df.apply(lambda x: get_first_half(x,df_data_sampled,'VO2'),axis=1)
    data_resume_cpet_df['first_half_HRSlope']=data_resume_cpet_df.apply(lambda x: get_first_half(x,df_data_sampled,'HR'),axis=1)
    data_resume_cpet_df['first_half_VCO2Slope']=data_resume_cpet_df.apply(lambda x: get_first_half(x,df_data_sampled,'VCO2'),axis=1)
    data_resume_cpet_df['first_half_VESlope']=data_resume_cpet_df.apply(lambda x: get_first_half(x,df_data_sampled,'VE'),axis=1)
    data_resume_cpet_df['first_half_RERSlope']=data_resume_cpet_df.apply(lambda x: get_first_half(x,df_data_sampled,'RER'),axis=1)
    data_resume_cpet_df['first_half_RRSlope']=data_resume_cpet_df.apply(lambda x: get_first_half(x,df_data_sampled,'RR'),axis=1)
    data_resume_cpet_df['first_half_O2Slope']=data_resume_cpet_df.apply(lambda x: get_first_half(x,df_data_sampled,'O2'),axis=1)
    data_resume_cpet_df['first_half_VEVCO2Slope']=data_resume_cpet_df.apply(lambda x: get_first_half(x,df_data_sampled,'VE/VCO2'),axis=1)
    data_resume_cpet_df['first_half_VEVO2Slope']=data_resume_cpet_df.apply(lambda x: get_first_half(x,df_data_sampled,'VE/VO2'),axis=1)
    # Second half
    data_resume_cpet_df['second_half_VO2Slope']=data_resume_cpet_df.apply(lambda x: get_second_half(x,df_data_sampled,'VO2'),axis=1)
    data_resume_cpet_df['second_half_HRSlope']=data_resume_cpet_df.apply(lambda x: get_second_half(x,df_data_sampled,'HR'),axis=1)
    data_resume_cpet_df['second_half_VCO2Slope']=data_resume_cpet_df.apply(lambda x: get_second_half(x,df_data_sampled,'VCO2'),axis=1)
    data_resume_cpet_df['second_half_VESlope']=data_resume_cpet_df.apply(lambda x: get_second_half(x,df_data_sampled,'VE'),axis=1)
    data_resume_cpet_df['second_half_RERSlope']=data_resume_cpet_df.apply(lambda x: get_second_half(x,df_data_sampled,'RER'),axis=1)
    data_resume_cpet_df['second_half_RRSlope']=data_resume_cpet_df.apply(lambda x: get_second_half(x,df_data_sampled,'RR'),axis=1)
    data_resume_cpet_df['second_half_O2Slope']=data_resume_cpet_df.apply(lambda x: get_second_half(x,df_data_sampled,'O2'),axis=1)
    data_resume_cpet_df['second_half_VEVCO2Slope']=data_resume_cpet_df.apply(lambda x: get_second_half(x,df_data_sampled,'VE/VCO2'),axis=1)
    data_resume_cpet_df['second_half_VEVO2Slope']=data_resume_cpet_df.apply(lambda x: get_second_half(x,df_data_sampled,'VE/VO2'),axis=1)
    if time_percent>40:
        # 15 to 85 percent
        data_resume_cpet_df['15_to_85_VO2Slope']=data_resume_cpet_df.apply(lambda x: get_slope_15_85(x,df_data_sampled,'VO2'),axis=1)
        data_resume_cpet_df['15_to_85_HRSlope']=data_resume_cpet_df.apply(lambda x: get_slope_15_85(x,df_data_sampled,'HR'),axis=1)
        data_resume_cpet_df['15_to_85_VCO2Slope']=data_resume_cpet_df.apply(lambda x: get_slope_15_85(x,df_data_sampled,'VCO2'),axis=1)
        data_resume_cpet_df['15_to_85_VESlope']=data_resume_cpet_df.apply(lambda x: get_slope_15_85(x,df_data_sampled,'VE'),axis=1)
        data_resume_cpet_df['15_to_85_RERSlope']=data_resume_cpet_df.apply(lambda x: get_slope_15_85(x,df_data_sampled,'RER'),axis=1)
        data_resume_cpet_df['15_to_85_RRSlope']=data_resume_cpet_df.apply(lambda x: get_slope_15_85(x,df_data_sampled,'RR'),axis=1)
        data_resume_cpet_df['15_to_85_O2Slope']=data_resume_cpet_df.apply(lambda x: get_slope_15_85(x,df_data_sampled,'O2'),axis=1)
        data_resume_cpet_df['15_to_85_VEVCO2Slope']=data_resume_cpet_df.apply(lambda x: get_slope_15_85(x,df_data_sampled,'VE/VCO2'),axis=1)
        data_resume_cpet_df['15_to_85_VEVO2Slope']=data_resume_cpet_df.apply(lambda x: get_slope_15_85(x,df_data_sampled,'VE/VO2'),axis=1)
        # First half
        data_resume_cpet_df['0_to_25_VO2Slope']=data_resume_cpet_df.apply(lambda x: get_first_quarter(x,df_data_sampled,'VO2'),axis=1)
        data_resume_cpet_df['0_to_25_HRSlope']=data_resume_cpet_df.apply(lambda x: get_first_quarter(x,df_data_sampled,'HR'),axis=1)
        data_resume_cpet_df['0_to_25_VCO2Slope']=data_resume_cpet_df.apply(lambda x: get_first_quarter(x,df_data_sampled,'VCO2'),axis=1)
        data_resume_cpet_df['0_to_25_VESlope']=data_resume_cpet_df.apply(lambda x: get_first_quarter(x,df_data_sampled,'VE'),axis=1)
        data_resume_cpet_df['0_to_25_RERSlope']=data_resume_cpet_df.apply(lambda x: get_first_quarter(x,df_data_sampled,'RER'),axis=1)
        data_resume_cpet_df['0_to_25_RRSlope']=data_resume_cpet_df.apply(lambda x: get_first_quarter(x,df_data_sampled,'RR'),axis=1)
        data_resume_cpet_df['0_to_25_O2Slope']=data_resume_cpet_df.apply(lambda x: get_first_quarter(x,df_data_sampled,'O2'),axis=1)
        data_resume_cpet_df['0_to_25_VEVCO2Slope']=data_resume_cpet_df.apply(lambda x: get_first_quarter(x,df_data_sampled,'VE/VCO2'),axis=1)
        data_resume_cpet_df['0_to_25_VEVO2Slope']=data_resume_cpet_df.apply(lambda x: get_first_quarter(x,df_data_sampled,'VE/VO2'),axis=1)
        # Second Quarter
        data_resume_cpet_df['25_to_50_VO2Slope']=data_resume_cpet_df.apply(lambda x: get_second_quarter(x,df_data_sampled,'VO2'),axis=1)
        data_resume_cpet_df['25_to_50_HRSlope']=data_resume_cpet_df.apply(lambda x: get_second_quarter(x,df_data_sampled,'HR'),axis=1)
        data_resume_cpet_df['25_to_50_VCO2Slope']=data_resume_cpet_df.apply(lambda x: get_second_quarter(x,df_data_sampled,'VCO2'),axis=1)
        data_resume_cpet_df['25_to_50_VESlope']=data_resume_cpet_df.apply(lambda x: get_second_quarter(x,df_data_sampled,'VE'),axis=1)
        data_resume_cpet_df['25_to_50_RERSlope']=data_resume_cpet_df.apply(lambda x: get_second_quarter(x,df_data_sampled,'RER'),axis=1)
        data_resume_cpet_df['25_to_50_RRSlope']=data_resume_cpet_df.apply(lambda x: get_second_quarter(x,df_data_sampled,'RR'),axis=1)
        data_resume_cpet_df['25_to_50_O2Slope']=data_resume_cpet_df.apply(lambda x: get_second_quarter(x,df_data_sampled,'O2'),axis=1)
        data_resume_cpet_df['25_to_50_VEVCO2Slope']=data_resume_cpet_df.apply(lambda x: get_second_quarter(x,df_data_sampled,'VE/VCO2'),axis=1)
        data_resume_cpet_df['25_to_50_VEVO2Slope']=data_resume_cpet_df.apply(lambda x: get_second_quarter(x,df_data_sampled,'VE/VO2'),axis=1)
        # Third quarter
        data_resume_cpet_df['50_to_75_VO2Slope']=data_resume_cpet_df.apply(lambda x: get_third_quarter(x,df_data_sampled,'VO2'),axis=1)
        data_resume_cpet_df['50_to_75_HRSlope']=data_resume_cpet_df.apply(lambda x: get_third_quarter(x,df_data_sampled,'HR'),axis=1)
        data_resume_cpet_df['50_to_75_VCO2Slope']=data_resume_cpet_df.apply(lambda x: get_third_quarter(x,df_data_sampled,'VCO2'),axis=1)
        data_resume_cpet_df['50_to_75_VESlope']=data_resume_cpet_df.apply(lambda x: get_third_quarter(x,df_data_sampled,'VE'),axis=1)
        data_resume_cpet_df['50_to_75_RERSlope']=data_resume_cpet_df.apply(lambda x: get_third_quarter(x,df_data_sampled,'RER'),axis=1)
        data_resume_cpet_df['50_to_75_RRSlope']=data_resume_cpet_df.apply(lambda x: get_third_quarter(x,df_data_sampled,'RR'),axis=1)
        data_resume_cpet_df['50_to_75_O2Slope']=data_resume_cpet_df.apply(lambda x: get_third_quarter(x,df_data_sampled,'O2'),axis=1)
        data_resume_cpet_df['50_to_75_VEVCO2Slope']=data_resume_cpet_df.apply(lambda x: get_third_quarter(x,df_data_sampled,'VE/VCO2'),axis=1)
        data_resume_cpet_df['50_to_75_VEVO2Slope']=data_resume_cpet_df.apply(lambda x: get_third_quarter(x,df_data_sampled,'VE/VO2'),axis=1)
        # Last quarter
        data_resume_cpet_df['75_to_100_VO2Slope']=data_resume_cpet_df.apply(lambda x: get_last_quarter(x,df_data_sampled,'VO2'),axis=1)
        data_resume_cpet_df['75_to_100_HRSlope']=data_resume_cpet_df.apply(lambda x: get_last_quarter(x,df_data_sampled,'HR'),axis=1)
        data_resume_cpet_df['75_to_100_VCO2Slope']=data_resume_cpet_df.apply(lambda x: get_last_quarter(x,df_data_sampled,'VCO2'),axis=1)
        data_resume_cpet_df['75_to_100_VESlope']=data_resume_cpet_df.apply(lambda x: get_last_quarter(x,df_data_sampled,'VE'),axis=1)
        data_resume_cpet_df['75_to_100_RERSlope']=data_resume_cpet_df.apply(lambda x: get_last_quarter(x,df_data_sampled,'RER'),axis=1)
        data_resume_cpet_df['75_to_100_RRSlope']=data_resume_cpet_df.apply(lambda x: get_last_quarter(x,df_data_sampled,'RR'),axis=1)
        data_resume_cpet_df['75_to_100_O2Slope']=data_resume_cpet_df.apply(lambda x: get_last_quarter(x,df_data_sampled,'O2'),axis=1)
        data_resume_cpet_df['75_to_100_VEVCO2Slope']=data_resume_cpet_df.apply(lambda x: get_last_quarter(x,df_data_sampled,'VE/VCO2'),axis=1)
        data_resume_cpet_df['75_to_100_VEVO2Slope']=data_resume_cpet_df.apply(lambda x: get_last_quarter(x,df_data_sampled,'VE/VO2'),axis=1)
        pass
    #data_resume_cpet_df['VTTime']=data_resume_cpet_df.apply(lambda x: get_vt_time(x,df_data_sampled),axis=1)
    data_resume_cpet_df['VTTime']=data_resume_cpet_df.apply(lambda x: get_vt_time(x,df_data_sampled),axis=1)
    data_resume_cpet_df['VO2atVT']=data_resume_cpet_df.apply(lambda x: get_VO2atVT(x,df_data_sampled),axis=1)
    data_resume_cpet_df['PeakVO2']=data_resume_cpet_df.apply(lambda x: get_highest_variable('VO2',x,df_data_sampled),axis=1)
    data_resume_cpet_df['PercentTimeAfterVT']=data_resume_cpet_df.apply(lambda x: get_percent_time_after_VT(x,df_data_sampled),axis=1)
    data_resume_cpet_df['VO2vsPeakVO2atVT'] = data_resume_cpet_df['VO2atVT']/data_resume_cpet_df['MaxVO2_EST']
    return data_resume_cpet_df
