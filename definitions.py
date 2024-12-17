from PIL import Image, ImageFont, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import csv
from sklearn.linear_model import LinearRegression
from scipy import stats
import itertools
import time
import pickle
import functools

def quad_tree_leaf_order(x:str, y:str)->int:
    if len(x) < len(y):
        return -1
    if len(x) > len(y):
        return 1
    if x == y: return 0
    return -1 if x < y else 1

def leaf_sorted(arr): 
    return sorted(arr,key=functools.cmp_to_key(quad_tree_leaf_order))

def combine_images(image_paths:list[str], save_to:str,labels:list[str], rows:int, columns:int, size_single:tuple[int,int]=(128,128),space_single:tuple[int,int] = (0,0), default_bg:tuple[int,int,int,int]=(0,0,0,255), font_color:tuple[int,int,int,int]=(255,255,255,255)):

    width, height = size_single
    swidth, sheight = space_single

    background_width = columns * (width + swidth) + swidth
    background_height = rows * (height + sheight) + sheight
    background = Image.new('RGBA', (background_width, background_height), default_bg)
    
    id = 0
    for y,x in itertools.product(range(rows), range(columns)):
        if id >= len(image_paths):
            break
        img = Image.open(image_paths[id])
        img = img.resize((width,height),resample=Image.BILINEAR)

        background.paste(img, (x*width + (x  + 1) * swidth, y*height + (y+1)*sheight))
        
        id += 1
        if id >= len(image_paths):
            break
    
    
    if len(labels) > 0 and len(labels) == len(image_paths):
        label_id = 0
        draw = ImageDraw.Draw(background)
        font = ImageFont.truetype("arial.ttf", 20)
        for y,x in itertools.product(range(rows), range(columns)):
            px = x * (width + swidth) + 0 * width + swidth
            py = (y) * (height + sheight) + 0.5 * sheight
            draw.text((px,py), labels[label_id], font_color, font=font)
            label_id += 1
        
    background.save(save_to)


def save_object(path:str, obj):
    remove_if_exists(path)
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp)

def load_object(path:str):
    with open(path, 'rb') as fp:
        return pickle.load(fp)

def save_txt(path:str, contents:str):
    remove_if_exists(path)
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(contents)
        
def append_save_txt(path:str, contents:str):
    with open(path, 'a', encoding='utf-8') as fp:
        fp.write(contents)

def load_txt(path:str):
    with open(path, 'r', encoding='utf-8') as fp:
        return fp.read()

def _is_power_of_2(n: int):
        return n != 0 and (n & (n-1) == 0)

def clamp(n, smallest, largest):
    return smallest if n < smallest else largest if n > largest else n

def map(v, f1, f2, t1, t2):
    return (v - f1) / (f2-f1) * (t2-t1) + t1

def normalize_2darray(arr2d):
    cp = np.copy(arr2d)
    for idx, x in np.ndenumerate(cp):
        cp[idx] = clamp(abs(round(x)), 0, 255)
    return np.uint8(cp)


def remove_if_exists(path):
    if os.path.isfile(path):
        #print(f"removing: {path}")
        os.remove(path)


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def to_hist_data(arr2d:np.ndarray):
    #r = np.int16(np.round(arr2d))
    r = arr2d
    return [x for idx, x in np.ndenumerate(r)]

def to_hist_dict(arr2d:np.ndarray)->dict[int,int]:
    d:dict[int,int] = dict()
    for _, v in np.ndenumerate(arr2d):
        iv = int(round(v))
        if iv not in d:
            d[iv] = 1
        else:
            d[iv] += 1
    return d



def read_image(path:str):
    img = Image.open(path).convert('L')
    #img = img.resize((8, 8), Image.Resampling.BILINEAR )
    #print("Readed", path, img.format, img.size, img.mode)
    return np.asarray(img)


def combine_image(ll:np.ndarray, lh:np.ndarray, hl:np.ndarray, hh:np.ndarray):
    s0 = ll.shape[0]
    s1 = ll.shape[1]
    res = np.zeros((s0*2, s1*2))

    res[:s0, :s1] = ll
    res[s0:, :s1] = lh
    res[:s0, s1:] = hl
    res[s0:, s1:] = hh

    return res


def sym_coord(x, len):
    new_x = x
    if new_x < 0:
        new_x = -new_x

    if new_x >= len:
        new_x %= 2*len-2
        if new_x >= len:
            new_x = 2*len-2-new_x

    return new_x


def n_scheme(N):
    if N == 1:
        return {
            "p": [1],
            "p_off": 0,
            "u": [0.5],
            "u_off": 0,
            "k": 1,
        }
    if N >= 2:
        n = N // 2
        p = [(-1)**(k+n-1) * np.prod([n+0.5-m for m in range(1, 2*n+1)])
             / (math.factorial(n-k)*math.factorial(n-1+k)*(k-0.5))
             for k in range(-n+1, n+1)]
        u = [v/2 for v in p]
        return {
            "p": p,
            "p_off": -n+1,
            "u": u,
            "u_off": -n+1,
            "k": 1,
        }

# for n in range(2,7,2): print(f"{n_scheme(n)['p']}{n_scheme(n)['u']}")


def lifting1d(arr1d, len, scheme):

    k = scheme["k"]
    p = scheme["p"]
    u = scheme["u"]
    p_off = scheme["p_off"]
    u_off = scheme["u_off"]
    hlen = len // 2

    ce = [arr1d[2*i] for i in range(hlen)]
    co = [arr1d[2*i+1] for i in range(hlen)]

    #print("arr1d", arr1d, "ce", ce, "co", co)

    d = [co[i] - round(sum([idv * ce[sym_coord(i-(p_off + id), hlen)]
                            for id, idv in enumerate(p)]))
         for i in range(hlen)]

    c = [ce[i] + round(sum([idv * d[sym_coord(i-(u_off + id), hlen)]
                            for id, idv in enumerate(u)]))
         for i in range(hlen)]

    # print("c",c,"d",d)

    c = [v*k for v in c]
    d = [v/k for v in d]

    return c, d


def path_to_range(path:str, all_c_count:int = 256) ->tuple[int,int]:
    if _is_all_C(path):
        return (0,all_c_count-1)
    radius = 2 ** (sum([1 if p == 'D' or p == 'E' else 2 if p == 'F' else 0 for p in path])-1) * (all_c_count - 1)
    radius = int(radius)
    return -radius, radius



def inv_lifting1d(arr1dl, arr1dh, len, scheme):
    k = scheme["k"]
    p = scheme["p"]
    u = scheme["u"]
    p_off = scheme["p_off"]
    u_off = scheme["u_off"]
    dlen = len * 2

    c = [v/k for v in arr1dl]
    d = [v*k for v in arr1dh]

    ce = [c[i] - round(sum([idv * d[sym_coord(i-(u_off + id), len)]
                            for id, idv in enumerate(u)]))
          for i in range(len)]

    co = [d[i] + round(sum([idv * ce[sym_coord(i-(p_off + id), len)]
                            for id, idv in enumerate(p)]))
          for i in range(len)]

    return [ce[i//2] if i % 2 == 0 else co[i//2] for i in range(dlen)]


def lifting2d(arr2d, shape, scheme):
    rows = np.array([lifting1d(arr2d[i], shape[1], scheme)
                    for i in range(shape[0])])

    l = rows[:, 0, :]
    h = rows[:, 1, :]

    l = l.transpose()
    h = h.transpose()

    l_cols = np.array([lifting1d(l[i], l.shape[1], scheme)
                      for i in range(l.shape[0])])
    h_cols = np.array([lifting1d(h[i], h.shape[1], scheme)
                      for i in range(h.shape[0])])

    ll = l_cols[:, 0, :].transpose()
    lh = l_cols[:, 1, :].transpose()
    hl = h_cols[:, 0, :].transpose()
    hh = h_cols[:, 1, :].transpose()

    return ll, lh, hl, hh


def inv_lifting2d(arr2dll, arr2dlh, arr2dhl, arr2dhh, shape, scheme):

    ll = arr2dll.transpose()
    lh = arr2dlh.transpose()
    hl = arr2dhl.transpose()
    hh = arr2dhh.transpose()

    l = np.array([inv_lifting1d(ll[i], lh[i], shape[0], scheme)
                 for i in range(shape[1])])
    h = np.array([inv_lifting1d(hl[i], hh[i], shape[0], scheme)
                 for i in range(shape[1])])

    l = l.transpose()
    h = h.transpose()

    return np.array([inv_lifting1d(l[i], h[i], l.shape[1], scheme) for i in range(shape[0]*2)])




def test_lifting_image(dir, filename, N, showres=True):
    path = f"{dir}/{filename}"
    savefolder = f"{dir}/res/{N}/"
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    img = read_image(path)
    scheme = n_scheme(N)
    ll, lh, hl, hh = lifting2d(img, img.shape, scheme)
    #origin_inv = inv_lifting2d(ll, lh, hl, hh, ll.shape, scheme)

    # dif_count = 0
    # for idx,v in np.ndenumerate(origin_inv):
    # if v != img[idx]: dif_count += 1

    # if dif_count>0: print(f"different pixels: {dif_count}")

    ll_hist = to_hist_data(ll)
    lh_hist = to_hist_data(lh)
    hl_hist = to_hist_data(hl)
    hh_hist = to_hist_data(hh)
    hists = [ll_hist, hl_hist, lh_hist, hh_hist]
    titles = ["LL", "HL", "LH", "HH"]
    min_range = [0, -127, -127, -127]
    max_range = [255, 127, 127, 127]

    whole = combine_image(ll, lh, hl, hh)
    whole = normalize_2darray(whole)

    img_filepath = f"{savefolder}{filename}_lift_N_{N}.png"
    hist_filepath = f"{savefolder}{filename}_lift_N_{N}_hist.png"

    whole_img = Image.fromarray(whole, mode="L")
    remove_if_exists(img_filepath)
    whole_img.save(img_filepath)
    #if showres: display(whole_img)

    f, a = plt.subplots(2, 2, dpi=300)
    a = a.ravel()
    for idx, ax in enumerate(a):
        ax.hist(hists[idx], range=(min_range[idx], max_range[idx]),
                bins=max_range[idx]-min_range[idx])
        ax.set_title(f"{filename} {titles[idx]} N={N}")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    plt.tight_layout()
    remove_if_exists(hist_filepath)
    plt.savefig(hist_filepath)
    if showres:
        plt.show()
    else:
        plt.close()

    print(f"{dir}/{filename} done. {img_filepath}, {hist_filepath}")


def _linear_reg(x, y):
    model = LinearRegression().fit(x.reshape((-1, 1)), y)
    return model.coef_[0], model.intercept_

# print(_linear_reg([1,2,3,4],[5,7,8,9]))


def _chi2_judge(data: dict, model, value_range: range):
    v_sum = sum(data.values())
    observed = [0 if i not in data else data[i] for i in value_range]
    expected = [math.exp(math.log(v_sum) + model.log_prob_of(i))
                for i in value_range]
    #print(f"v_sum: {v_sum}")
    #print(f"observed: {observed}")
    #print(f"expected: {expected}")
    #for i in range(len(observed)):
        #print(i, observed[i], expected[i])
    tol = (v_sum - sum(expected)) / (value_range.stop-value_range.start)
    expected = list(map(lambda v: v + tol, expected))

    return stats.chisquare(observed, expected)


def img2d_to_count_kv(arr2d:np.ndarray) -> dict:
    data = {}
    for row in arr2d:
        for pixel in row:
            ip = int(pixel)
            data[ip] = data.get(ip, 0) + 1
    return data

def _is_all_C(path):
    for c in path:
        if c != 'C':
            return False
    return True


def _decompose_r(arr2d, shape, scheme, max_depth, root_path="", res={}):
    if len(root_path) >= max_depth:
        return
    ll, lh, hl, hh = lifting2d(arr2d, shape, scheme)
    for d in [["C", ll], ["D", lh], ["E", hl], ["F", hh]]:
        cur_root = root_path + d[0]
        res["".join(cur_root)] = d[1]
        _decompose_r(d[1], d[1].shape, scheme, max_depth, cur_root, res)


def _order_to_path(r, c, depth):
    res = ""
    for d in range(depth):
        ri = r % 2
        ci = c % 2
        cur_res = ""
        if ri == 0:
            cur_res = "C" if ci == 0 else "E"
        else:
            cur_res = "D" if ci == 0 else "F"
        res = cur_res + res
        r //= 2
        c //= 2
    return res


def _path_to_order(path):
    res_r = 0
    res_c = 0
    step = 1
    for c in reversed(path):
        if c == 'C':
            res_r += step * 0
            res_c += step * 0
        elif c == 'D':
            res_r += step * 1
            res_c += step * 0
        elif c == 'E':
            res_r += step * 0
            res_c += step * 1
        elif c == 'F':
            res_r += step * 1
            res_c += step * 1
        step *= 2
    return res_r, res_c




def apply_to_average(average_cache, decomposed):
    for k, v in decomposed.items():
        if k not in average_cache:
            average_cache[k] = {}
        d = average_cache[k]

        ints = to_hist_data(v)
        for i in ints:
            int_i = int(round(i))
            if int_i not in d:
                d[int_i] = int(0)
            d[int_i] = round(d[int_i]+1)


def fit_test():
    return


def construct_image_tree(img_arr2d, depth, scheme):
    res = {}
    res[""] = img_arr2d
    for cd in range(depth):
        for path_group in itertools.product(['C', 'D', 'E', 'F'], repeat=cd):
            path = "".join(path_group)
            i = res[path]
            c,d,e,f = lifting2d(i, i.shape, scheme)
            res[path + "C"] = c
            res[path + "D"] = d
            res[path + "E"] = e
            res[path + "F"] = f
    return res

def set_boundary(img2d: np.ndarray, v:int):
    if not img2d.flags.writeable:
        img2d = img2d.copy()
    for x in range(0, img2d.shape[1]):
        img2d[0][x] = v
        img2d[-1][x] = v
    for y in range(0, img2d.shape[0]):
        img2d[y][0] = v
        img2d[y][-1] = v
    return img2d

def _r_decompose_image_structure(path:str,img_tree:dict[str,np.ndarray],leaf_set:set[str],boundary:bool=True)->np.ndarray:
    if path in leaf_set:
        img = img_tree[path]
        if boundary:
            img = set_boundary(img,255)
        else:
            img = img.copy()
        return img
    C = _r_decompose_image_structure(path + "C", img_tree, leaf_set, boundary)
    D = _r_decompose_image_structure(path + "D", img_tree, leaf_set, boundary)
    E = _r_decompose_image_structure(path + "E", img_tree, leaf_set, boundary)
    F = _r_decompose_image_structure(path + "F", img_tree, leaf_set, boundary)
    return combine_image(C,D,E,F)
    


def decompose_image_structure(img_tree:dict[str,np.ndarray],leaf_set:set[str],boundary:bool=True)->np.ndarray:
    return normalize_2darray(_r_decompose_image_structure("",img_tree,leaf_set,boundary))

def load_csv(dir):
    csv_path = f"{dir}/total.csv"
    res = {}
    with open(csv_path, 'r', newline='') as f:
        for row in csv.reader(f):
            if row[0] not in res:
                res[row[0]] = {}
            res[row[0]][int(row[1])] = int(row[2])
    return res


def deep_decompose_to_kv(dir, filename, depth=1):
    path = f"{dir}/{filename}"
    if not os.path.isfile(path):
        print(f"Image at {path} not found. ")
        return

    savefolder = f"{dir}/decompose/{filename}"

    img = read_image(path)
    scheme = n_scheme(1)
    size = img.shape[0]

    res = {}

    _decompose_r(img, img.shape, scheme, depth, "", res)

    return res

def save_image(path, arr2d:np.ndarray):
    whole_img = Image.fromarray(arr2d, mode="L")
    remove_if_exists(path)
    whole_img.save(path)

def save_j2k_image(path, arr2d:np.ndarray):
    whole_img = Image.fromarray(arr2d, mode="L")
    remove_if_exists(path)
    whole_img.save(path, format="JPEG2000", irreversible = False)

def get_j2k_codeword_len(arr2d:np.ndarray)->int:
    path = "__TEMP__cache__img__.j2k"
    save_j2k_image(path, arr2d)
    size = os.path.getsize(path)
    remove_if_exists(path)
    return size

def get_datetime_string()->str:
    return time.strftime('%Y%m%d-%H%M%S')


def visualize_quadtree(leaf: set[str], max_depth:int)->np.ndarray:
    img = np.zeros((2 ** max_depth,2 ** max_depth))
    img_tree = construct_image_tree(img, max_depth, n_scheme(1))
    return normalize_2darray(decompose_image_structure(img_tree, leaf, True))

def test_deep_decompose(dir, filename, depth=1, average_cache=None, lock=None):

    path = f"{dir}/{filename}"
    if not os.path.isfile(path):
        print(f"Image at {path} not found. ")
        return

    savefolder = f"{dir}/decompose/{filename}"

    img = read_image(path)
    scheme = n_scheme(1)
    size = img.shape[0]

    res = {}

    _decompose_r(img, img.shape, scheme, depth, "", res)

    apply_to_average(average_cache, res)

    return
    cell_size = size
    cell_1d_num = 1
    img_cache = np.zeros((size, size))
    for d in range(depth):
        cell_size //= 2
        cell_1d_num *= 2

        #fig = plt.figure(figsize=(256 * (2 ** d), 256*(2 ** d)))

        for r in range(cell_1d_num):
            for c in range(cell_1d_num):
                ri = r * cell_size
                ci = c * cell_size
                #is_L = True if ri==0 and ci==0 else False
                path = _order_to_path(r, c, d+1)
                data = res[path]

                img_cache[ri:ri+cell_size, ci:ci+cell_size] = data[:, :]

                #ax = fig.add_subplot(cell_1d_num,cell_1d_num,1 + c * cell_1d_num + r)
                #ax.hist(to_hist_data(data), range=(0 if is_L else -128,256 if is_L else 128),bins=256)
                # ax.set_title(f"{path}")
                # ax.set_xlabel("value")
                # ax.set_ylabel("count")

        img_folder = f"{savefolder}/img/combined"
        img_path = f"{img_folder}/{d+1}.png"
        create_folder_if_not_exists(img_folder)
        remove_if_exists(img_path)
        Image.fromarray(normalize_2darray(img_cache), mode="L").save(img_path)

        #hist_folder = f"{savefolder}/img/combined"
        #hist_path = f"{hist_folder}/{d+1}_hist.png"
        # create_folder_if_not_exists(hist_folder)
        # remove_if_exists(hist_path)
        # plt.tight_layout()
        # plt.savefig(hist_path)
        # plt.close()

    for k, v in res.items():
        hist_folder = f"{savefolder}/hist/{len(k)}"
        #img_folder = f"{savefolder}/img/{len(k)}"

        create_folder_if_not_exists(hist_folder)
        # create_folder_if_not_exists(img_folder)

        hist_filepath = f"{hist_folder}/{k}.png"
        #img_filepath = f"{img_folder}/{k}.png"

        is_H = not _is_all_C(k)

        # save image
        # remove_if_exists(img_filepath)
        #Image.fromarray(normalize_2darray(v), mode="L").save(img_filepath)

        # save histogram
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(to_hist_data(v), range=(-128 if is_H else 0,
                128 if is_H else 256), bins=256)
        ax.set_title(f"{filename} {k}")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        plt.tight_layout()
        remove_if_exists(hist_filepath)
        plt.savefig(hist_filepath)
        plt.close()

from scipy.special import digamma, polygamma

def digamma_inv(y:float)->float:
    x = 0
    if y >= -2.22:
        x = np.exp(y) + 0.5
    else:
        x = -1.0/(y - digamma(1))

    for _ in range(5):
        x -= (digamma(x)-y)/polygamma(1,x)
    return x

def get_E_ln_theta(alpha:float,beta:float)->tuple[float,float]:# E[ln(theta)], E[ln(1-theta)]
    s = digamma(alpha + beta)
    return digamma(alpha) - s, digamma(beta) - s
