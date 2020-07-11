import csv
import random
import numpy as np 
import functions


'''
    交叉
'''
def crossover(data, num_parents, solutions, bias):
    parents = functions.choose_roulette(data, num_parents, solutions, bias)
    child = []
    for child_index in range(len(parents[0])):
        parent_vector = []
        for parents_index in range(len(parents)):
            parent_vector.append(parents[parents_index][child_index])
        child.append(functions.XLM(parent_vector))

    return child

'''
    新しい世代の作成
'''
def make_children(data, num_parents, elite_preservation_rate, solutions, bias):
    children = []
    evaluations = functions.get_evaluations_list(data, solutions, bias)
    rank_list = functions.get_ranking_list(evaluations)
    # エリート保存分
    for order in range(1, int(elite_preservation_rate * len(data) + 1)):
        for index in range(len(rank_list)):
            if rank_list[index] == order:
                children.append(data[index])
                break
    
    for element in range(len(data) - int(len(data) * elite_preservation_rate)):
        children.append(crossover(data, num_parents, solutions, bias))

    return children

'''
    Main
'''
# エリート保存率
elite_preservation_rate = 0.05
# 一度の交叉で使う親の数
num_parents = 2
# 読み込むファイル
read_filename = 'pre_experiment/mock_random_matrix_10'
# 書き込むファイル
write_filename = 'dressing/children'
# 実行回数
num_execute = 250

# 局所解ファイル
solutions_file = 'pre_experiment/mock_solutions_100'
# 評価結果のファイル
result_file = 'dressing/evaluation_result'

# 局所解ファイルの読み込み
solutions_data = functions.read_csv(solutions_file)
del solutions_data[0]
solutions_data = functions.transform_to_float(solutions_data)

# 局所解とバイアスに分ける
solutions, bias = functions.divide_solutions_bias(solutions_data)

# 評価値の結果のリスト
evaluations_result = []

for num_experiment in range(1 , 3501):
    print(num_experiment)
    # 対象のデータの読み込み
    data = functions.read_csv(read_filename)
    del data[0]
    data = functions.transform_to_float(data)
    # 次の世代の作成
    for num in range(num_execute):
        print(num)
        data = make_children(data, num_parents, elite_preservation_rate, solutions, bias)
        

    # 新しい世代をcsvに書き込む
    functions.write_csv(write_filename + '_%i' % num_experiment, data)

    evaluations = functions.get_evaluations_list(data, solutions, bias)
    evaluation_vector = functions.get_result(data, evaluations, num_experiment, functions.get_best_solution_index(bias), solutions)
    evaluations_result.append(evaluation_vector)
final_result = functions.get_final_result(evaluations_result)

functions.write_result(result_file, evaluations_result, final_result)
