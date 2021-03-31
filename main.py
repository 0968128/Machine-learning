import numpy as np

student_number = "0968128"
server_url = "https://programmeren9.cmgt.hr.nl:9000/"
no_cache = False

def extract_from_json_as_np_array(key, json_data):
    """ helper functie om data uit de json te halen en om te zetten naar numpy array voor sklearn"""
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)