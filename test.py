# import numpy as np

# X_train_path = "/Users/chouchou/chou/github/ai4life/X.txt"
# # X_test_path = file_path_testx

# y_train_path = "/Users/chouchou/chou/github/ai4life/dataY.txt"
# # y_test_path = file_path_testy

# n_steps = 32 # 32 timesteps per series
# n_categories = 22

# # Load the networks inputs

# def load_X(X_path, n_steps=32):
#     file = open(X_path, 'r')
#     X_ = np.array(
#         [elem for elem in [
#             row.split(',') for row in file
#         ]],
#         dtype=np.float32
#     )
#     file.close()
#     blocks = int(len(X_) / n_steps)

#     X_ = np.array(np.split(X_,blocks))

#     return X_

# # Load the networks outputs
# def load_y(y_path):
#     file = open(y_path, 'r')
#     y_ = np.array(
#         [elem for elem in [
#             row.replace('  ', ' ').strip().split(' ') for row in file
#         ]],
#         dtype=np.int32
#     ).reshape(-1)
#     file.close()

#     # for 0-based indexing
#     return y_

# print(load_X("/Users/chouchou/chou/github/ai4life/X.txt").shape)


# X_train = load_X(X_train_path)
# # X_test = load_X(X_test_path)

# y_train = load_y(y_train_path)
# # y_test = load_y(y_test_path)

# print(X_train.shape)
# # for x in X_train:
# #     # print(x.shape)
# #     string = ""
# #     string2 = ""
# #     string3 = ""
# #     for xx in x:
# #         xx_add1 = xx + 1
# #         xx_minus1 = xx - 1
# #         for i in xx:  
# #           string += f'{i},'
# #         string = string[:-1]+"\n"
# #         for i in xx_add1:  
# #           string2 += f'{i},'
# #         string2 = string2[:-1]+"\n"
# #         for i in xx_minus1:  
# #           string3 += f'{i},'
# #         string3 = string3[:-1]+"\n"
# #     with open(f'X.txt', 'a') as file:
# #       file.writelines(string)
# #       file.writelines(string2)
# #       file.writelines(string3)

# # for x in y_train:
# #     for i in x:
# #       with open(f'Y.txt', 'a') as file:
# #         file.writelines(str(i)+"\n")
# #         file.writelines(str(i)+"\n")
# #         file.writelines(str(i)+"\n")'



file_pathx = "dataX.txt"
file_pathy = "dataY.txt"
file_path_trainx = "dataX_train.txt"
file_path_trainy = "dataY_train.txt"

file_path_testx = "dataX_test.txt"
file_path_testy = "dataY_test.txt"

n_steps = 32
split_ratio = 0.9
read_filesx = open(file_pathx, "r").readlines()
print(len(read_filesx))
read_filesy = open(file_pathy, "r").readlines()
print(len(read_filesy))
Y = {}
for read_file in read_filesy:
    read_file = read_file.strip()
    if read_file not in Y:
        Y[read_file] = 1
    else:
        Y[read_file] += 1
t = 0
for key in Y:
    for i in range(0, Y[key]):
        if (i <= int(Y[key]*(1 - split_ratio))):
            with open(file_path_testy, "a") as file:
                file.write(key + "\n")
            for j in range(0, n_steps):
                with open(file_path_testx, "a") as file:
                    file.write(read_filesx[t])
                t += 1
        else:
            with open(file_path_trainy, "a") as file:
                file.write(key + "\n")
            for j in range(0, n_steps):
                with open(file_path_trainx, "a") as file:
                    file.write(read_filesx[t])
                t += 1