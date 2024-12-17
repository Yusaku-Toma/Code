from lib import definitions as defs
from lib import SNSGHyperGenModel_Toma as snsghypergen
from lib import WPTGenModel_Toma as wptgen
import math

from PIL import Image


# 木の最大深さ2^max_depth = size (2^6=64), 画像サイズは64×64ピクセル
max_depth = int(6)

# 画像のピクセルのビット数, 変える必要なしとの記述
bit_count = int(8)

# 変換基底 Haar = defs.n_scheme(1), 変える必要なしとの記述
scheme = defs.n_scheme(1)
Q = 10

# 初期パラメータの値, 画像データ生成の段階ではとりあえず既知としておく
alpha_g = 0.8 * Q
beta_g = Q - alpha_g
alpha_b = 0.3 * Q
beta_b = Q - alpha_b
g_of_C = 0.9
g_other= 0.7

# 各ノードにSNSGモデルを割り当てた初期化
def _gen_node_initer(node:wptgen.NodeData,path:str) -> snsghypergen.SNSGHyperGenModel:
    leaf_model = snsghypergen.SNSGHyperGenModel()
    global alpha_g, beta_g, alpha_b, beta_b
    leaf_model.alpha_g = alpha_g
    leaf_model.beta_g = beta_g
    leaf_model.alpha_b = alpha_b
    leaf_model.beta_b = beta_b
    return leaf_model

# 初期化実行
wpt_gen = wptgen.WPTGenModel(max_depth, bit_count, _gen_node_initer)

# gの初期値の設定
for node,path in wpt_gen.tree.dfs():
    depth = len(path)
    if depth < max_depth: # 最大深さに達していない場合
        if defs._is_all_C(path): # 低周波数領域の場合
            node.data.log2gs = math.log2(g_of_C)
            node.data.zero_gs = False
        else: # その他周波数領域の場合
            node.data.log2gs = math.log2(g_other)
            node.data.zero_gs = False   
    else: # 一番深いノードのgは0
        node.data.log2gs = -1
        node.data.zero_gs = True

# 画像を生成
img = wpt_gen.generate()

# 画像結果の出力(数値)
print(img)

output_path = "output_image.png"  # 保存するファイル名
img_uint8 = Image.fromarray(img)  # NumPy 配列を画像データに変換
img_uint8.save(output_path)       # ファイルに保存

# 画像結果の出力(png)
print(f"Image saved to {output_path}")