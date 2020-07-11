import csv
import numpy as np 
import functions

'''
    +1か-1をそれぞれ1/2の確率で出力する
'''
def noise():
    return 1 if np.random.rand() >= 0.5 else -1

'''
    ランダムベクトルの生成
    要素は全て[-1, 1]の範囲
'''
def make_random_vector(num_dimentions):
    random_vector = []
    for dimention in range(num_dimentions):
        random_vector.append(np.random.rand() * noise())

    return random_vector

'''
    ランダム行列の生成
    要素は全て[-1, 1]の範囲
'''
def make_random_matrix(num_elements, num_dimensions):
    random_matrix = []
    for element in range(num_elements):
        random_matrix.append(make_random_vector(num_dimensions))
    return random_matrix

'''
'''
def take_top_100(data, evaluations, ranking_list, solutions, bias):
    new_data = []
    new_ranking = []
    for element in range(len(data)):
        if ranking_list[element] <= 100:
            new_data.append(data[element])
            new_ranking.append(ranking_list[element])
    new_evaluations = functions.get_evaluations_list(new_data, solutions, bias)
    
    return new_data, new_evaluations, new_ranking

'''
    Main
'''
# 潜在空間の次元数
num_dimentions = 100
# 局所解ファイル
solutions_file = 'pre_experiment/mock_solutions_100'
# 評価結果のファイル
result_file = 'Random/evaluation_result'

# 局所解ファイルの読み込み
solutions_data = functions.read_csv(solutions_file)
del solutions_data[0]
solutions_data = functions.transform_to_float(solutions_data)

# 局所解とバイアスに分ける
solutions, bias = functions.divide_solutions_bias(solutions_data)
solutions = np.array(solutions)
bias = np.array(bias)

evaluations_result = []

for num_experiment in range(1, 101):
    print(num_experiment)
    random = make_random_matrix(5000, 100)

    # 評価値の結果のリスト
    evaluations = functions.get_evaluations_list(random, solutions, bias)
    rankings = functions.get_ranking_list(evaluations)
    matrix100, evaluations100, rankings100 = take_top_100(random, evaluations, rankings, solutions, bias)

    evaluation_vector = functions.get_result(matrix100, evaluations100, num_experiment, functions.get_best_solution_index(bias), solutions)
    evaluations_result.append(evaluation_vector)

final_result = functions.get_final_result(evaluations_result)
functions.write_result(result_file, evaluations_result, final_result)
