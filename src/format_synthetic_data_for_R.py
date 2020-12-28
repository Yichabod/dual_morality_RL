import pandas as pd

model_generated = pd.read_csv("/Users/maxlangenkamp/Desktop/UROP/dual_morality_RL/randomized_100_score_dif_9_5.csv")
score_dif = []
time_constraint = []

time_x_distrib = []
in_distrib = []
userid = []
grid_num = []
for index, row in model_generated.iterrows():
    if (row["grid_num"] > 100) and not (200 < row["grid_num"] < 300):
        score_dif.append(row["reward"]-row["best_reward"])
        grid_num.append(row["grid_num"])
        if row["model"] == 'delay':
            time = 0.5
        else:
            time = -0.5
        if row["grid_num"] in [101,102,103,104,105,106,107,108]:
            distrib = 0.5 #out of distrib
        if row["grid_num"] > 300:
            distrib = -0.5 #in distrib
        time_constraint.append(time)
        in_distrib.append(distrib)
        time_x_distrib.append(time*distrib)
        userid.append(row["id"])
    
d = {'score_dif': score_dif, 'time_constraint': time_constraint, 'in_distrib': in_distrib, 'time_x_distrib':time_x_distrib, 'userid': userid, 'gridnum': grid_num}

mlm_data = pd.DataFrame(d)
mlm_data.to_csv('../data/cleaned_data_model_new.csv')
