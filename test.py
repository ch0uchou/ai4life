# import numpy as np

# X_train_path = "/Users/chouchou/chou/github/ai4life/X.txt"
# # X_test_path = file_path_testx

# y_train_path = "/Users/chouchou/chou/github/ai4life/dataY.txt"
# # y_test_path = file_path_testy

# n_steps = 32 # 32 timesteps per series
# n_categories = 22

# # Load the networks inputs

# def load_X(X_path):
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
#     )
#     file.close()

#     # for 0-based indexing
#     return y_ - 1

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
# #         file.writelines(str(i)+"\n")