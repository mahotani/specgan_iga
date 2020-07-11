import csv
import numpy as np
import functions

'''
    交叉
'''
def crossover(parents, num_parents, num_dimentions):
    child = np.zeros(num_dimentions)
    for child_index in range(num_dimentions):
        parents_vector = np.zeros(num_parents)
        for parents_index in range(len(parents)):
            parents_vector[parents_index] = parents[parents_index, child_index]
            # parents_vector.append(parents[parents_index][child_index])
        child[child_index] = functions.XLM(parents_vector)
        # child.append(functions.XLM(parents_vector))

    return child

'''
    MGGによる次世代の生成
'''
def next_generation_MGG(data, solutions, bias, num_parents, num_children, num_elite_preservation, num_dimentions):
    
    parents, parents_index = functions.random_parent(data, num_parents)
    #print(parents_index)
    '''
    for index in range(len(parents_index)):
        count = 0
        
        for index2 in range(index):
            if parents_index[index2] < parents_index[index]:
                count += 1
        del data[parents_index[index] - count]
    '''

    children = np.zeros((num_children + num_parents, num_dimentions))

    for parent_index in range(len(parents)):
        children[parent_index] = np.copy(parents[parent_index])

    #print(children)
    for cross in range(num_children):
        child_vector = crossover(parents, num_parents, num_dimentions)
        children[num_parents + cross] = np.copy(child_vector)

    #print(children)

    evaluations = functions.get_evaluations_list(children, solutions, bias)
    rank_list = functions.get_ranking_list(evaluations)
    # print(rank_list)
    # print(evaluations)
    '''
    # 上位のものを元のリストに戻す
    for order in range(1, num_elite_preservation + 1):
        for index in range(len(rank_list)):
            if rank_list[index] == order:
                data[parents_index[order-1]] = np.copy(children[index])
                
                break
    
    # print(functions.get_evaluations_list(children, solutions, bias))
    # ルーレット選択により選ばれた個体を一つ戻す
    selected = functions.choose_roulette(children, (num_parents - num_elite_preservation), solutions, bias)
    for element in range(len(selected)):
        data[parents_index[num_elite_preservation + element]] = np.copy(selected[element])
    '''
    min_index1 = np.argmin(evaluations[0:4])
    data[parents_index[0]] = np.copy(children[min_index1])

    min_index2 = np.argmin(evaluations[4:8]) + 4
    data[parents_index[1]] = np.copy(children[min_index2])

    min_index3 = np.argmin(evaluations[8:12]) + 8
    data[parents_index[2]] = np.copy(children[min_index3])

    min_index4 = np.argmin(evaluations[12:16]) + 12
    data[parents_index[3]] = np.copy(children[min_index4])
    return data


'''
    Main
'''
# エリート保存数
num_elite_preservation = 1
# 一度の交叉で使う親の数
num_parents = 4
# 一度の交叉で生まれる子の数
num_children = 12
# 読み込むファイル
read_filename = 'pre_experiment/mock_random_matrix_100'
# 書き込むファイル
write_filename = 'MGG/children'
# 実行回数
num_execute = int(5000 / 12)
# 潜在空間の次元数
num_dimentions = 100
# 局所解ファイル
solutions_file = 'pre_experiment/mock_solutions_100'
# 評価結果のファイル
result_file = 'MGG/evaluation_result'


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
    # before_index = functions.get_result(data, functions.get_evaluations_list(data, solutions, bias), num_experiment, functions.get_best_solution_index(bias), solutions)[2]
    # 次の世代の作成
    for num in range(num_execute):
        
        # print(functions.get_evaluation_value(data[before_index], solutions, bias))
        data = next_generation_MGG(data, solutions, bias, num_parents, num_children, num_elite_preservation, num_dimentions)

        # if num % 50 == 0:  
        #print('-------')
        #print(functions.get_result(data, functions.get_evaluations_list(data, solutions, bias), num_experiment, functions.get_best_solution_index(bias), solutions))
        
        # before_index = functions.get_result(data, functions.get_evaluations_list(data, solutions, bias), num_experiment, functions.get_best_solution_index(bias), solutions)[2]
    # 新しい世代をcsvに書き込む
    functions.write_csv(write_filename + '_%i' % num_experiment, data)

    evaluations = functions.get_evaluations_list(data, solutions, bias)
    evaluation_vector = functions.get_result(data, evaluations, num_experiment, functions.get_best_solution_index(bias), solutions)
    evaluations_result.append(evaluation_vector)
final_result = functions.get_final_result(evaluations_result)

functions.write_result(result_file, evaluations_result, final_result)
