from scipy.stats import bernoulli, beta, geom


# 高周波成分が従うSNSG分布をモデル化するクラス
class SNSGHyperGenModel:

    # 各変数名称はこれまでの表記通り
    def __init__(self, alpha_g=2, beta_g=2, alpha_b=2, beta_b=2):
        self.alpha_g = alpha_g
        self.beta_g = beta_g
        self.alpha_b = alpha_b
        self.beta_b = beta_b
        # あとでBeta分布で生成するが仮置き
        self.theta_g = 0.5
        self.theta_b = 0.5

    # θをBeta分布で生成
    # random=Falseにして引数を指定すれば特定のθを設定可能
    def init_params(self, random=True, theta_g=None, theta_b=None):
        if random:
            self.theta_g = beta.rvs(self.alpha_g, self.beta_g)
            self.theta_b = beta.rvs(self.alpha_b, self.beta_b)
        else:
            self.theta_g = theta_g if theta_g is not None else self.theta_g
            self.theta_b = theta_b if theta_b is not None else self.theta_b

    # SNSG分布による出力生成
    def generate(self) -> float:

        # θの範囲確認
        if not (0 <= self.theta_g <= 1):
            raise ValueError("theta_g must be in the range [0, 1].")
        if not (0 <= self.theta_b <= 1):
            raise ValueError("theta_b must be in the range [0, 1].")


        # θ_bの確率で0を生成
        is_zero = bernoulli.rvs(self.theta_b) == 1
        if is_zero:
            return 0

        # 0生成でない場合, 符号を同様に確からしく決定→絶対値をパラメータθ_gの幾何分布で決定
        is_minus = bernoulli.rvs(0.5) == 0
        l = geom.rvs(self.theta_g)

        return -l if is_minus else l


    def plot_histogram(self, num_samples=10000, bins=11, hist_range=(-5, 5), output_file="snsghypergen-res.png"):
        import matplotlib.pyplot as plt
        samples = [self.generate() for _ in range(num_samples)]
        fig, ax = plt.subplots()
        ax.hist(samples, bins=bins, range=hist_range)
        ax.set_title("SNSGHyperGenModel Histogram")
        ax.set_xlabel("Generated Values")
        ax.set_ylabel("Frequency")
        fig.savefig(output_file)
        plt.close()


# このコードを単体で実行したときのテストとしてのplot_histogram実行
if __name__ == "__main__":
    model = SNSGHyperGenModel()
    model.init_params(random=True)
    model.plot_histogram()