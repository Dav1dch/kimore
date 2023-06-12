# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import numpy as np
import glob

data_path = "kimoreData"

def read_pose_simulate_data(data_path):
    file_list_1 = os.listdir(data_path)
    file_list_1 = sorted(file_list_1)
    csv_file_num = 0
    data_list = []

    for file_folder_1 in file_list_1:
        full_file_path_1 = os.path.join(data_path, file_folder_1)
        file_list_2 = os.listdir(full_file_path_1)
        file_list_2 = sorted(file_list_2)

        for file_folder_2 in file_list_2:
            full_file_path_2 = os.path.join(full_file_path_1, file_folder_2)
            #print(full_file_path_2)
            remove_file_list = os.listdir(full_file_path_2)

            for remove_file in remove_file_list:
                if "align" in remove_file:
                    remove_path = os.path.join(full_file_path_2, remove_file)
                    os.remove(remove_path)

            robot_simulate_name = "datarehab_" + str(csv_file_num) + "_0_*.csv"
            robot_pose_name = "joint_pose_*.csv"

            robot_simulate_file = glob.glob(os.path.join(full_file_path_2, robot_simulate_name))
            human_pose_file = glob.glob(os.path.join(full_file_path_2, robot_pose_name))
            #print(human_pose_file)
            csv_file_num += 2

            if file_folder_2 == "Es5" and file_folder_1 == "P_ID7":
                continue

            if "90" in file_folder_1 or "91" in file_folder_1:
                human_pose_file.sort(key=lambda x: list(x[36:-4]))

            else:
                human_pose_file.sort(key=lambda x: list(x[35:-4]))

            #print(human_pose_file)
            data_list_single = human_pose_file + robot_simulate_file
            data_list.append(data_list_single)

    return data_list

def write_to_txt(data_list):
    txt_file_name = ""
    with open("train_file.txt", "w") as f:
        for path_list in data_list:
            write_line  =  " ".join(item for item in path_list)
            f.write(write_line + "\n")
    f.close()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_list = read_pose_simulate_data(data_path)

    write_to_txt(data_list)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
