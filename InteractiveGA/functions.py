import csv
import numpy as np 
import math

'''
    csvファイルの読み込み
'''
def read_csv(filename):
    with open(filename, 'r') as csv_files:
        data = list(csv.reader(csv_files))

    return data

'''
    csvファイルへの書き込み
'''
def write_csv(filename, data):
    with open(filename, 'w') as csv_files:
        writer = csv.writer(csv_files)
        for row in data:
            writer.writerow(row)

'''
    csvファイルへの書き込み（ベクトル用）
'''
def write_csv_for_vector(filename, vector):
    with open(filename, 'w') as csv_files:
        writer = csv.writer(csv_files)
        writer.writerow(vector)

'''
    2次元配列の全要素をfloat型に変換
'''
def transform_to_float(data):
    new_array = []
    for row in data:
        new_vector = []
        for column in row:
            new_vector.append(float(column))
        new_array.append(new_vector)
    
    return new_array

'''
    2次元配列の全要素をint型に変換
'''
def transform_to_int(data):
    new_array = []
    for row in data:
        new_vector = []
        for column in row:
            new_vector.append(int(column))
        new_array.append(new_vector)
    
    return new_array

'''
    ランダムな数のリストを作成
'''
def make_random_list(max_value, num_element):
    if max_value < num_element:
        return None

    random_list = []
    for index in range(num_element):
        random_value = np.random.randint(max_value)
        while random_value in random_list:
            random_value = np.random.randint(max_value)
        random_list.append(random_value)
    
    return random_list

'''
    REX
'''
def REX(parent_vector):
    weight = np.mean(parent_vector)
    child_element = weight
    for element in parent_vector:
        child_element += (element - weight) * np.random.normal(0, math.sqrt(2 / len(parent_vector)))
    if child_element > 1:
        child_element = 1.0
    if child_element < -1:
        child_element = -1.0
        
    return child_element

'''
    交叉（子の個体を生成）
'''
def crossover(parents):
    child = np.zeros(len(parents[0]))
    for child_index in range(len(parents[0])):
        parents_vector = np.zeros(len(parents))
        for parents_index in range(len(parents)):
            parents_vector[parents_index] = parents[parents_index][child_index]
        child[child_index] = REX(parents_vector)

    return child

'''
    子の個体群の生成
'''
def make_children(parents, num_children):
    children = np.zeros((num_children, len(parents[0])))
    for child in range(num_children):
        new_child = crossover(parents)
        children[child] = np.copy(new_child)
    
    return children

'''
    親が入っていた部分に子のベクトルを差し替える
'''
def replace(data, children, parents_index, children_index):
    data_copy = data
    for i in range(len(parents_index)):
        data_copy[parents_index[i]] = children[children_index[i]]
    
    return data_copy
