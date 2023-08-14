import csv
import os

import numpy as np

import os
import csv

import os
import csv


def extract_time(date_time):
    # 将时间戳字符串按空格进行分割
    hours = np.array([])
    minutes = np.array([])
    seconds = np.array([])
    date_time = date_time.split(" ")
    # 提取出日期部分和时间部分
    time_part = date_time[1]  # "0:00:00"
    # 将时间部分再按冒号进行分割
    time_components = time_part.split(":")
    # 分别提取出时、分、秒
    hour = int(time_components[0])  # 0
    minute = int(time_components[1])  # 0
    second = int(time_components[2])  # 0

    # 将提取的时分秒组合成新的时间字符串，格式为 "0:00:00"
    new_time = f"{hour}:{minute}:{second}"
    return new_time


def process_csv_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                data = list(csv_reader)
                print()
            for row in data:
                if "Time" in row:
                    print(row["Time"])
                    row["Time"] = extract_time(row["Time"])
                    print(row["Time"])

            with open(file_path, 'w', newline='', encoding='utf-8') as file:
                fieldnames = csv_reader.fieldnames
                csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
                csv_writer.writeheader()
                csv_writer.writerows(data)


# 指定要处理的文件夹路径
folder_path = "datas/new2"
process_csv_files_in_folder(folder_path)

print("Time 列替代完成！")
