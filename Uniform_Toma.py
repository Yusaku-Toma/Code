import math
import numpy as np

# 前提として, 画像各ピクセルがとる値は整数

# 低周波成分が従う一様分布をモデル化するクラス
class Uniform:
    value_min = 0
    value_max = 2 # 可変？

    # 分布を学習するメソッド？ 未実装のまま, 詳細不明
    def learn(self, arr2d: np.ndarray, power: float = 1, scale: float = 1):
        pass
        
    # 与えられた行列nが一様分布に従うときの出現確率(の底2対数)計算
    def log2_prob_of(self, n: np.ndarray):
        if not isinstance(n, np.ndarray):
            raise TypeError("Input 'n' must be a numpy ndarray.")
        
        try:
            res = 0
            v = -math.log2(self.value_max - self.value_min + 1)
            for e in np.nditer(n):
                ei = int(e)
                if ei >= self.value_min and ei <= self.value_max:
                    res += v # 足し算なのは対数をとっているから, この方が計算安定
                else:
                    res += -np.inf # 範囲外のnの成分には-∞の対数=ほぼ0の出現確率を返す
            return res
        
        except Exception as e:
            print(f"An error occurred: {e}")
        raise
    
    # 同じことを自然対数で行列arr2dに対して実行, 用途不明
    def va_E_ln_p(self, arr2d: np.ndarray):
        res = 0
        v = -math.log(self.value_max - self.value_min + 1)
        for e in np.nditer(arr2d):
            ei = int(e)
            if ei >= self.value_min and ei <= self.value_max:
                res += v
            else:
                res += -np.inf
        return res
        
    # 行列nの理想符号長を計算
    def code_length_of(self, n: np.ndarray):
        return -self.log2_prob_of(n)
    
    # value_minとvalue_maxを辞書形式で保存
    def save_dict(self) -> dict:
        return {"value_min": self.value_min, "value_max": self.value_max}
    
    # 辞書から2つを読み込んでクラスに適用
    def load_dict(self, data: dict):

        # 2つのデータ不正に対する例外処理
        try:
            if "value_min" not in data or "value_max" not in data: # 存在しているか
                raise KeyError("The dictionary must contain 'value_min' and 'value_max'.")
            if not isinstance(data["value_min"], (int, float)) or not isinstance(data["value_max"], (int, float)): # 値がfloatか
                raise TypeError("'value_min' and 'value_max' must be integers or floats.")
            if data["value_min"] > data["value_max"]: # min>maxになってないか
                raise ValueError("'value_min' must be less than or equal to 'value_max'.")
            
            self.value_min = data["value_min"]
            self.value_max = data["value_max"]
        
        except KeyError as ke:
            print(f"KeyError: {ke}")
        except TypeError as te:
            print(f"TypeError: {te}")
        except ValueError as ve:
            print(f"ValueError: {ve}")
