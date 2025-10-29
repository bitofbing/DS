import pickle
with open('cv_results\cheng_li_yuan_shiyan\sonar_final_results.pkl', 'rb') as file:
    data = pickle.load(file)
    print(data)