import os
import pandas as pd


# c = len(os.listdir(filefoldnames))
# for filefoldname in os.listdir(filefoldnames):
#     filefold = os.path.join(filefoldnames, filefoldname)
#     file = os.listdir(filefold)
#     path = os.path.join(filefold, file[0])
#
#     # checking if file exist or not
#     if (os.path.isfile(path)):
#
#         # os.remove() function to remove the file
#         os.remove(path)
#
#         # Printing the confirmation message of deletion
#         print("File Deleted successfully")
#     else:
#         print("File does not exist")
#     # Showing the message instead of throwig an error
#

filefoldnames = r'G:\train'
no_training = []
for filefoldname in os.listdir(filefoldnames):
    filefold = os.path.join(filefoldnames, filefoldname)
    count = len(os.listdir(filefold))
    if count == 6:
        no_training.append(filefoldname)


