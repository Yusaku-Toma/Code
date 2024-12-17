import itertools
import queue
from typing import Generator
from typing import Any

class Node:
    parent = None
    C = None
    D = None
    E = None
    F = None
    expard_mark = False
    data=None

    def __init__(self, parent=None) -> None:
        self.parent = parent

    def is_expand(self):
        cur_node = self
        while cur_node is not None:
            if not cur_node.expard_mark:
                return False
            cur_node = cur_node.parent
        return True

    def expand(self):
        if self.C == None:
            self.C = Node(self)

        if self.D == None:
            self.D = Node(self)

        if self.E == None:
            self.E = Node(self)

        if self.F == None:
            self.F = Node(self)

        self.expard_mark = True

    def shrink(self):
        self.expard_mark = False

    def dfs_children(self):
        yield self
        if self.C is not None:
            yield from self.C.dfs_children()
        if self.D is not None:
            yield from self.D.dfs_children()
        if self.E is not None:
            yield from self.E.dfs_children()
        if self.F is not None:
            yield from self.F.dfs_children()

    def bfs_children(self):
        q = queue.Queue()
        q.put(self)
        while not q.empty():
            n = q.get()
            yield n

            if n.C is not None:
                q.put(n.C)
            if n.D is not None:
                q.put(n.D)
            if n.E is not None:
                q.put(n.E)
            if n.F is not None:
                q.put(n.F)
    
    def get_child(self, path:str):
        n = len(path)
        if n == 0: return self
        if path[0] == 'C':
            return self.C.get_child(path[1:])
        
        if path[0] == 'D':
            return self.D.get_child(path[1:])
        
        if path[0] == 'E':
            return self.E.get_child(path[1:])
        
        if path[0] == 'F':
            return self.F.get_child(path[1:])
        





class QuadTree:
    head:Node = None

    def __init__(self, depth=0) -> None:
        self.head = Node()
        q = queue.Queue()
        q.put((self.head, 0))
        while not q.empty():
            qele = q.get()
            node = qele[0]
            ndepth = qele[1]
            nextdepth = ndepth + 1
            node.expand()
            if nextdepth >= depth:
                continue
            q.put((node.C, nextdepth))
            q.put((node.D, nextdepth))
            q.put((node.E, nextdepth))
            q.put((node.F, nextdepth))

    def _r_clone(self, node: Node, ref_node: Node):
        if node.is_expand():
            ref_node.expand()
            self._r_clone(node.C, ref_node.C)
            self._r_clone(node.D, ref_node.D)
            self._r_clone(node.E, ref_node.E)
            self._r_clone(node.F, ref_node.F)

    def clone(self):
        res = QuadTree()
        self._r_clone(self.head, res.head)
        return res

    def _r_leaf_paths(self, node: Node, path: str):
        if node.is_expand():
            yield from self._r_leaf_paths(node.C, path + "C")
            yield from self._r_leaf_paths(node.D, path + "D")
            yield from self._r_leaf_paths(node.E, path + "E")
            yield from self._r_leaf_paths(node.F, path + "F")
        else:
            yield path

    def leaf_paths(self) -> Generator[str, Any, Any]:
        yield from self._r_leaf_paths(self.head, "")


    def _r_inner_paths(self, node: Node, path: str):
        if node.is_expand():
            yield path
            yield from self._r_inner_paths(node.C, path + "C")
            yield from self._r_inner_paths(node.D, path + "D")
            yield from self._r_inner_paths(node.E, path + "E")
            yield from self._r_inner_paths(node.F, path + "F")

    def inner_paths(self) -> Generator[str, Any, Any]:
        yield from self._r_inner_paths(self.head, "")

    def binary_expression(self):
        return tuple([1 if node.is_expand() else 0 for node in self.head.bfs_children()])
    
    def get_node_from_path(self, path:str):
        return self.head.get_child(path)
    
    def _r_get_node_paths(self, node:Node, cur_path:str, data:set):
        data.add(cur_path)

        if not node.is_expand(): return

        self._r_get_node_paths(node.C, cur_path + "C", data)
        self._r_get_node_paths(node.D, cur_path + "D", data)
        self._r_get_node_paths(node.E, cur_path + "E", data)
        self._r_get_node_paths(node.F, cur_path + "F", data)
        

    def get_node_paths(self):
        res = set()
        self._r_get_node_paths(self.head,"", res)
        return res
    
    def bfs(self) -> Generator[tuple[Node, str], Any, Any]:
        q = queue.Queue[tuple[Node,str]]()
        q.put((self.head,""))
        while not q.empty():
            node, path = q.get()
            yield (node, path)
            if node.C is not None:
                q.put((node.C, path+"C"))
                q.put((node.D, path+"D"))
                q.put((node.E, path+"E"))
                q.put((node.F, path+"F"))
    
    def _r_dfs(self, node:Node, path:str) -> Generator[tuple[Node, str], Any, Any]:
        if node.C is not None:
            yield from self._r_dfs(node.C, path+"C")
            yield from self._r_dfs(node.D, path+"D")
            yield from self._r_dfs(node.E, path+"E")
            yield from self._r_dfs(node.F, path+"F")
        yield (node, path)

    def dfs(self) -> Generator[tuple[Node, str], Any, Any]:
        yield from self._r_dfs(self.head, "")




    
    



'''def All_QuadTrees_Leaves(max_depth: int):
    tree = QuadTree(max_depth)
    searched = {}
    nodes = list(tree.head.bfs_children())
    inner_count = int((4**(max_depth)-1)//3)
    # print(inner_count)
    for binary in itertools.product([0, 1], repeat=inner_count):
        r_bin = tuple(reversed(binary))
        for i in range(len(r_bin)):
            if r_bin[i] == 0:
                nodes[i].shrink()
            else:
                nodes[i].expand()

        exp = tree.binary_expression()
        # print(exp)
        if exp in searched:
            continue
        searched[exp] = None
        yield tuple(tree.leaf_paths())'''


def _r_All_QuadTrees_Leaves(cur_depth: int, max_depth: int,cur_path:str):
    next_depth = cur_depth + 1
    yield {cur_path}
    if(cur_depth < max_depth):
        for comb in itertools.product(_r_All_QuadTrees_Leaves(next_depth, max_depth, cur_path + "C"),
                                  _r_All_QuadTrees_Leaves(next_depth, max_depth, cur_path + "D"),
                                  _r_All_QuadTrees_Leaves(next_depth, max_depth, cur_path + "E"),
                                  _r_All_QuadTrees_Leaves(next_depth, max_depth, cur_path + "F")
                                  ):
            yield comb[0] | comb[1] | comb[2] | comb[3]
        
    


def All_QuadTrees_Leaves(max_depth: int):
    yield from _r_All_QuadTrees_Leaves(0, max_depth,"")


def Leaves_To_Inner(leaves: dict):
    res = {''}

    for k in leaves:
        for e in range(1,len(k)):
            res.add(k[:e])
    return res

def Leaves_To_Tree(leaves: dict):
    return Leaves_To_Inner(leaves) | leaves


def Generate_Float_Tree(floats: dict, depth:int):
    tree = QuadTree(depth)
    for k,v in floats.items():
        n = tree.get_node_from_path(k)
        n.data = v
    return tree

