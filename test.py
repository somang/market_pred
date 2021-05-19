# class Node:
    
#     def __init__(self, str_name):
#         self.list_child_objects = []
#         self.str_name = str_name
        
#     def __repr__(self):
#         return "<Node '{}'>".format(self.str_name)
    
#     def print_all(self):
#         print(self)
#         for child in self.list_child_objects:
#             child.print_all()

#     # Create the append function so that the STDOUT matches the output in the problem explanation
#     def append(self, node):
#         ## this is just a binary tree... why don't we use left and right obj instead...
        
#         if len(self.list_child_objects) < 3:
#             self.list_child_objects.append(node)
        
        
# if __name__ == '__main__':
#     root_object = Node('root')
#     child1_object = Node('child1')
#     child2_object = Node('child2')
#     child3_object = Node('child3')
    
#     root_object.append(child1_object)
#     root_object.append(child2_object)
#     child1_object.append(child3_object)
    
#     root_object.print_all()
# # read the string filename
# filename = 'i.txt'
 
# # Using readlines()
# file1 = open(filename, 'r')
# Lines = file1.readlines()

# D = {}
# # Using readlines()
# for line in Lines:
#     hostname = line.split(' ')[0]
#     # unicomp6.unicomp.net - - [01/Jul/1995:00:00:06 -0400] "GET /shuttle/countdown/ HTTP/1.0" 200 3985
#     if hostname in D.keys():
#         D[hostname] = D[hostname] + 1
#     else:
#         D[hostname] = 1

# sortedD = sorted(D.keys(), key=lambda x:x.lower())

# L = []
# for k in sortedD:
#     logline = k + " " + str(D[k])
#     L.append(logline)
#     # print(logline)
    
# ofname = "records_"+filename
# print(ofname)
# outfile = open(ofname, 'w')
# for l in L:
#     outfile.write(l + '\n')

# outfile.close()


#!/bin/python3

import math
import os
import random
import re
import sys
def hacker_cards(collection, d):
    cards_to_buy = []
    c = 1
    while (d > 0):
        print(c, d, collection)
        if c not in collection:
            if (c <= d):
                cards_to_buy.append(c)
            d -= c

        c += 1
    return cards_to_buy
    
    
if __name__ == '__main__':


    collection = [4, 6, 12, 8]

    print(hacker_cards(collection, 14))
