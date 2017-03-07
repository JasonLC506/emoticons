"""
binary heap
March 07, 2017
"""

class biheap(object):
    def __init__(self):
        self.heap = []
        self.length = 0

    def buildheap(self, list_unsorted):
        self.length = len(list_unsorted)
        