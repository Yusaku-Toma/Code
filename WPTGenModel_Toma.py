from . import TreeTools as treetools
from . import SNSGHyperGenModel_Toma as snsghypergen
from . import UniformGenModel as unigen
import math
from . import definitions as defs
import numpy as np
from scipy.stats import bernoulli as ber
from typing import Callable
from collections import deque

# ノードデータを保持するクラス
class NodeData:
    model = None
    log2gs:float = -1
    zero_gs:bool = False

    def __init__(self, log2gs, zero_gs):
        self.log2gs = log2gs
        self.zero_gs = zero_gs

    # g^sを出力
    def gs(self)->float:
        if self.zero_gs: return 0
        return 2 ** self.log2gs

# 全ノードの初期化
def _default_node_initer(node:NodeData,path:str) -> snsghypergen.SNSGHyperGenModel:
    leaf_model = snsghypergen.SNSGHyperGenModel()
    leaf_model.alpha_g = 1
    leaf_model.beta_g = 1
    leaf_model.alpha_b = 1
    leaf_model.beta_b = 1
    return leaf_model

class WPTGenModel:
    tree:treetools.QuadTree = None
    _max_depth:int = 1
    _pixel_bit:int = 8
    _scheme = None
    _width:int
    _height:int


    def __init__(self, max_depth:int=9, pixel_bit:int=8, leaf_model_initer:Callable[[NodeData,str],object] = _default_node_initer):
        
        self._width = 2 ** max_depth
        self._height = 2 ** max_depth
        self._max_depth = max_depth
        self._pixel_bit = pixel_bit
        self.tree = treetools.QuadTree(self._max_depth)
        self._scheme = defs.n_scheme(1)

        for node, path in self.tree.dfs():
            nd = NodeData(-1, False)
            if len(path) == self._max_depth:
                nd.zero_gs = True
            # 低周波成分のセットアップ
            if defs._is_all_C(path):
                uni = unigen.UniformGenModel()
                uni.value_min = 0
                uni.value_max = int((2 ** pixel_bit) - 1)
                nd.model = uni
            # 高周波成分のセットアップ
            else:
                nd.model = leaf_model_initer(node, path)
            node.data = nd

    # 全ノードデータを辞書形式で出力
    def get_tree_models(self) -> dict[str, NodeData]:
        res = {}

        for node, path in self.tree.dfs():
            res[path] = node.data

        return res
    
    # 非再帰的な幅優先探索による木の生成
    def _r_init_tree(self, node: treetools.Node, path: str):
        queue = deque([(node, path)])
        while queue:
            current_node, current_path = queue.popleft()
            if len(current_path) >= self._max_depth:
                current_node.shrink()
                continue
            if ber.rvs((2 ** current_node.data.log2gs) if not current_node.data.zero_gs else 0) == 1:
                current_node.expand()
                queue.append((current_node.C, current_path + "C"))
                queue.append((current_node.D, current_path + "D"))
                queue.append((current_node.E, current_path + "E"))
                queue.append((current_node.F, current_path + "F"))
            else:
                current_node.shrink()

    def init_params(self, init_tree:bool=True, init_params:bool=True):
        if init_tree:
            self._r_init_tree(self.tree.head, "")
        if init_params:
            for node, path in self.tree.dfs():
                node.data.model.init_params()


    # 逐次的なデータ生成 (処理速度高速化)
    def _r_generate(self, node: treetools.Node, path: str, keep_structure: bool = False) -> np.ndarray:
        if node.is_expand():
            c = self._r_generate(node.C, path + "C", keep_structure)
            d = self._r_generate(node.D, path + "D", keep_structure)
            e = self._r_generate(node.E, path + "E", keep_structure)
            f = self._r_generate(node.F, path + "F", keep_structure)
            return defs.inv_lifting2d(c, d, e, f, c.shape, self._scheme) if not keep_structure else defs.combine_image(c, d, e, f)
        depth = len(path)
        size = 2 ** (self._max_depth - depth)
        img2d = np.empty(shape=(size, size))
        model = node.data.model

        for idx, _ in np.ndenumerate(img2d):
            img2d[idx] = model.generate()

        return img2d

            
    # 再帰的に_r_generateを呼び出し, 生成データを正規化して保存
    def generate(self, keep_structure:bool = False):
        res = self._r_generate(self.tree.head, "", keep_structure)
        max_pix_v = int((2 ** self._pixel_bit) - 1)
        for idx, x in np.ndenumerate(res):
            res[idx] = int(defs.clamp(abs(round(x)), 0, max_pix_v))
        return defs.normalize_2darray(res)
        
