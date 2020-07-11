import csv
import numpy as np 
import functions

'''
    交叉
'''
def crossover(parents, num_parents, num_dimentions):
    child = np.zeros(num_dimentions)
    for child_index in range(len(parents[0])):
        parents_vector = np.zeros(num_parents)
        for parents_index in range(len(parents)):
            parents_vector[parents_index] = parents[parents_index][child_index]
            # parents_vector.append(parents[parents_index][child_index])
        child[child_index] = functions.XLM(parents_vector)
        # child.append(functions.XLM(parents_vector))

    return child

'''
    JGG+による次世代の生成
'''
def next_generation_JGG(data, solutions, bias, num_parents, num_children, num_dimentions):
    new_generation = np.zeros((len(data), num_dimentions))
    
    turn = 0
    while len(data) != 0:
        parents, parents_index = functions.random_parent(data, num_parents)
        
        for index in range(len(parents_index)):
            count = 0
            for index2 in range(index):
                if parents_index[index2] < parents_index[index]:
                    count += 1
            data = np.delete(data, parents_index[index] - count, axis=0)
            # del data[parents_index[index] - count]

        children = np.zeros((num_children, num_dimentions))

        for cross in range(num_children):
            child_vector = crossover(parents, num_parents, num_dimentions)
            children[cross] = np.copy(child_vector)
            # children.append(child_vector)

        evaluations = np.array(functions.get_evaluations_list(children, solutions, bias))
        rank_list = np.array(functions.get_ranking_list(evaluations))
        
        '''
        # 上位のものを元のリストに戻す
        for order in range(1, num_parents+1):
            for index in range(len(rank_list)):
                if rank_list[index] == order:
                    new_generation[turn * num_parents + order - 1] = np.copy(children[index])
                    # new_generation.append(children[index])
                    # del children[index]
                    # del rank_list[index]
                    break 
        '''
        min_index1 = np.argmin(evaluations[0:4])
        new_generation[turn * num_parents] = np.copy(children[min_index1])

        min_index2 = np.argmin(evaluations[4:8]) + 4
        new_generation[turn * num_parents + 1] = np.copy(children[min_index2])

        '''
        min_index3 = np.argmin(evaluations[8:12]) + 8
        new_generation[turn * num_parents + 2] = np.copy(children[min_index3])

        min_index4 = np.argmin(evaluations[12:16]) + 12
        new_generation[turn * num_parents + 3] = np.copy(children[min_index4])
        '''
        turn += 1
        
    return new_generation

'''
    Main
'''
# 一度の交叉で使う親の数
num_parents = 2
# 一度の交叉で生まれる子の数
num_children = 8
# 潜在空間の次元数
num_dimentions = 100
# 読み込むファイル
read_filename = 'pre_experiment/mock_random_matrix_100'
# 書き込むファイル
write_filename = 'JGGimprove/children'
# 実行回数
num_execute = int(5000/300)

# 局所解ファイル
solutions_file = 'pre_experiment/mock_solutions_100'
# 評価結果のファイル
result_file = 'JGGimprove/evaluation_result'

# 局所解ファイルの読み込み
solutions_data = functions.read_csv(solutions_file)
del solutions_data[0]
solutions_data = functions.transform_to_float(solutions_data)

# 局所解とバイアスに分ける
solutions, bias = functions.divide_solutions_bias(solutions_data)
solutions = np.array(solutions)
bias = np.array(bias)

# 評価値の結果のリスト
evaluations_result = []

for num_experiment in range(1, 101):
    print(num_experiment)
    # 対象のデータの読み込み
    data = functions.read_csv(read_filename)
    del data[0]
    data = functions.transform_to_float(data)
    data = np.array(data)
    for num in range(num_execute):

        evaluations = functions.get_evaluations_list(data, solutions, bias)
        print(functions.get_result(data, evaluations, num_experiment, functions.get_best_solution_index(bias), solutions))
        data = next_generation_JGG(data, solutions, bias, num_parents, num_children, num_dimentions)
        evaluations = functions.get_evaluations_list(data, solutions, bias)
        
        
    functions.write_csv(write_filename + '_%i' % num_experiment, data)

    evaluations = functions.get_evaluations_list(data, solutions, bias)
    evaluations_vector = functions.get_result(data, evaluations, num_experiment, functions.get_best_solution_index(bias), solutions)
    evaluations_result.append(evaluations_vector)

final_result = functions.get_final_result(evaluations_result)
functions.write_result(result_file, evaluations_result, final_result)
