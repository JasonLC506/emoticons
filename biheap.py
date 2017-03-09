"""
binary heap
March 07, 2017
"""

class BiHeap(object):
    """
    with zero-index, default minimal heap
    supporting tuple heap with key
    """
    def __init__(self):
        self.heap = []  # [[key_value, pointer_originlist]]
        self.length = 0
        self.minial = True
        self.key = None
        self.originlist = {}
        self.maxpointer = None
        self.revmap = {}

    def buildheap(self, list_unsorted, minimal = True, key = None, identifier = None):
        """
        :param list_unsorted:
        :param minimal: True for minimal heap, False for maximal heap
        :param key: key field for comparison
                    for diction or list input, the index can be string or int;
                    default using whole input
        :param identifier: identifier for each item
        """
        self.length = len(list_unsorted)
        self.minimal = minimal
        self.originlist = {pointer: list_unsorted[pointer] for pointer in range(self.length)}
        self.maxpointer = self.length - 1
        self.key = key
        self.identifier = identifier
        self.heap = [[self.itemkeyvalue(self.originlist[index]), index] for index in range(self.length)]
        self.revmap = {self.itemidentifier(self.originlist[index]): index for index in range(self.length)}
        for p in range(self.length/2):
            pindex = self.length/2 - p -1
            self.downsort(pindex)
        return self

    def find(self, identifier):
        if identifier in self.revmap.keys():
            return self.revmap[identifier]
        else:
            return None

    def fetch(self, index):
        if index >= self.length:
            return None
        else:
            return self.originlist[self.pointer(self.heap[index])]

    def pop(self):
        item = self.fetch(0)
        self.delete(0)
        return item

    def insert(self, item):
        self.maxpointer += 1
        self.originlist[self.maxpointer] = item
        self.length += 1
        self.heap.append([self.itemkeyvalue(item),self.maxpointer])
        self.revmap[self.itemidentifier(item)] = self.length-1
        self.upsort(self.length - 1)
        return self

    def delete(self, index):
        if index is None:
            return self
        if index >= self.length:
            print self.revmap
            print self.originlist
            print self.heap
            raise ValueError("index out of range")
        temp = self.heap[index]
        del self.revmap[self.itemidentifier(self.fetch(index))]
        self.heap[index] = self.heap[self.length - 1]
        self.revmap[self.itemidentifier(self.fetch(index))] = index
        self.downsort(index)
        del self.heap[self.length -1]
        self.length += -1

        del self.originlist[self.pointer(temp)]

        return self

    def update(self, index, item):
        olditem = self.fetch(index)
        if olditem is None:
            raise ValueError("index out of range")
        else:
            self.originlist[self.pointer(self.heap[index])] = item
            oldvalue = self.itemkeyvalue(olditem)
            newvalue = self.itemkeyvalue(item)
            self.heap[index][0] = newvalue # heap structure dependent
            if self.compare(newvalue, oldvalue):
                self.upsort(index)
            else:
                self.downsort(index)
        return self

    def itemkeyvalue(self, item):
        if self.key is None:
            return item
        else:
            return item[self.key]

    def itemidentifier(self, item):
        if self.identifier is None:
            return item
        else:
            return item[self.identifier]

    def keyvalue(self, heapitem):
        return heapitem[0] # heap structure dependent

    def pointer(self, heapitem):
        return heapitem[1] # heap structure dependent

    def upsort(self, cindex):
        pindex = self.parentindex(cindex)
        if pindex is None:
            return self
        if not self.compare(self.keyvalue(self.heap[pindex]), self.keyvalue(self.heap[cindex])):
            temp = self.heap[cindex]
            self.heap[cindex] = self.heap[pindex]
            self.heap[pindex] = temp
            self.revmap[self.itemidentifier(self.fetch(cindex))] = cindex
            self.revmap[self.itemidentifier(self.fetch(pindex))] = pindex
            self.upsort(pindex)
        return self

    def downsort(self, pindex):
        lc = self.childindex(pindex, lr = False)
        rc = self.childindex(pindex, lr = True)
        if lc is None:
            return self
        if rc is None:
            optiitem, optiindex = self.heap[lc], lc
        else:
            prefer = self.compare(self.keyvalue(self.heap[lc]), self.keyvalue(self.heap[rc]))
            if prefer:
                optiitem, optiindex = self.heap[lc], lc
            else:
                optiitem, optiindex = self.heap[rc], rc
        preferp = self.compare(self.keyvalue(self.heap[pindex]), self.keyvalue(optiitem))
        if not preferp:
            self.heap[optiindex] = self.heap[pindex]
            self.heap[pindex] = optiitem
            self.revmap[self.itemidentifier(self.fetch(pindex))] = pindex
            self.revmap[self.itemidentifier(self.fetch(optiindex))] = optiindex
            self.downsort(optiindex)
        return self

    def compare(self, v1, v2):
        if self.minimal:
            if v1 <= v2:
                return True
            else:
                return False
        else:
            if v1 >= v2:
                return True
            else:
                return False

    def childindex(self, pindex, lr = False):
        """
        find child index
        :param pindex: parent index
        :param lr: False for left, True for right
        :return: child_index
        """
        if lr:
            child_index = (pindex + 1) * 2
        else:
            child_index = pindex * 2 + 1
        if child_index >= self.length:
            return None
        else:
            return child_index

    def parentindex(self, cindex):
        if cindex > 0:
            return int(cindex - 1)/2
        else:
            return None

if __name__ == "__main__":
    h = BiHeap()
    h.buildheap([[7,2,3,4],[2,1,1,1],[4,0,1,0],[5,3,4,2]], key=2, identifier = 0)
    h.insert([1,2,9,0])
    h.delete(0)
    h.update(0,[3,7,5,9])
    print h.originlist
    print h.heap
    print h.revmap

    h2 = BiHeap().buildheap([2,3,4,1,5,7], minimal=False)
    h2.insert(-1)
    print h2.originlist
    print h2.heap


    print h2.find(-2)