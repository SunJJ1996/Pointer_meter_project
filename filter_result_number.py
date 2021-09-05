import cv2
import numpy as np
import math


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass


def count_distence(x1, y1, x2, y2):
    result = math.sqrt(
        math.pow(
            x1 -
            x2,
            2) +
        math.pow(
            y1 -
            y2,
            2))
    return result


def filter_edge(results, img_x, img_y):
    rusults_NoEdge = []
    img_h = img_y * 2
    img_w = img_x * 2
    for result in results:
        x, y, w, h = cv2.boundingRect(result.bbox_list)
        if (img_w * 0.9 > x > img_w * 0.1) and (img_h * 0.9 > y > img_h * 0.1):
            rusults_NoEdge.append(result)
    return rusults_NoEdge


def filter_if_has_number(results):
    result_onlyNum = []
    for result in results:
        if is_number(result.content) is True:
            result_onlyNum.append(result)
    return result_onlyNum


def filter_height_Zscore(results):
    result_suitable_high = []
    data = []
    for result in results:
        x, y, w, h = cv2.boundingRect(result.bbox_list)
        data.append(h)
    data_mean = np.mean(data)
    data_std = np.std(data, ddof=1)
    for result in results:
        x, y, w, h = cv2.boundingRect(result.bbox_list)
        z_score = (h - data_mean) / data_std
        # print(z_score)
        if abs(z_score) < 1:
            result_suitable_high.append(result)
    return result_suitable_high


def filter_height_mean(results):
    result_suitable_high = []
    data = []
    for result in results:
        x, y, w, h = cv2.boundingRect(result.bbox_list)
        data.append(h)
    if len(data) >= 3:
        data.remove(max(data))
        data.remove(min(data))
        data_mean = np.mean(data)
        for result in results:
            x, y, w, h = cv2.boundingRect(result.bbox_list)
            if data_mean * 1.35 > h > data_mean * 0.65:
                result_suitable_high.append(result)
    else:
        result_suitable_high = results
    return result_suitable_high


def filter_distence_Zscore(results, core_coordinate_x, core_coordinate_y):
    results_suitable_distence = []
    data_distence = []
    for result in results:
        x, y, w, h = cv2.boundingRect(result.bbox_list)
        distence_temp = count_distence(x, y, int(core_coordinate_x), int(core_coordinate_y))
        data_distence.append(distence_temp)
    data_mean = np.mean(data_distence)
    data_std = np.std(data_distence, ddof=1)
    for result in results:
        x, y, w, h = cv2.boundingRect(result.bbox_list)
        distence = count_distence(x, y, int(core_coordinate_x), int(core_coordinate_y))
        z_score = (distence - data_mean) / data_std
        if abs(z_score) < 1.5:
            results_suitable_distence.append(result)
    return results_suitable_distence


def filter_area_Zscore(results):
    results_suitable_area = []
    data_distence = []
    for result in results:
        x, y, w, h = cv2.boundingRect(result.bbox_list)
        area_temp = w * h
        data_distence.append(area_temp)
    data_mean = np.mean(data_distence)
    data_std = np.std(data_distence, ddof=1)
    for result in results:
        x, y, w, h = cv2.boundingRect(result.bbox_list)
        area = w * h
        z_score = (area - data_mean) / data_std
        if abs(z_score) < 1.5:
            results_suitable_area.append(result)
    return results_suitable_area


# 凸包//////////////////////////
def get_leftbottompoint(p):
    k = 0
    for i in range(1, len(p)):
        if p[i]['y'] < p[k]['y'] or (p[i]['y'] == p[k]['y'] and p[i]['x'] < p[k]['x']):
            k = i
    return k


# 叉乘计算方法
def multiply(p1, p2, p0):
    return (p1['x'] - p0['x']) * (p2['y'] - p0['y']) - (p2['x'] - p0['x']) * (p1['y'] - p0['y'])


# 获取极角，通过求反正切得出，考虑pi / 2的情况
def get_arc(p1, p0):
    # 兼容sort_points_tan的考虑
    if (p1['x'] - p0['x']) == 0:

        if ((p1['y'] - p0['y'])) == 0:
            return -1;
        else:
            return math.pi / 2

    tan = float((p1['y'] - p0['y'])) / float((p1['x'] - p0['x']))
    arc = math.atan(tan)
    if arc >= 0:
        return arc
    else:
        return math.pi + arc


# 对极角进行排序
def sort_points_tan(p, k):
    p2 = []
    for i in range(0, len(p)):
        p2.append({"index": i, "arc": get_arc(p[i], p[k])})
    p2.sort(key=lambda k: (k.get('arc', 0)))
    p_out = []
    for i in range(0, len(p2)):
        p_out.append(p[p2[i]["index"]])
    return p_out


def graham_scan(p):
    k = get_leftbottompoint(p)
    p_sort = sort_points_tan(p, k)

    p_result = [None] * len(p_sort)
    p_result[0] = p_sort[0]
    p_result[1] = p_sort[1]
    p_result[2] = p_sort[2]

    top = 2
    for i in range(3, len(p_sort)):
        # 叉乘为正则符合条件
        while (top >= 1 and multiply(p_sort[i], p_result[top], p_result[top - 1]) > 0):
            top -= 1
        top += 1
        p_result[top] = p_sort[i]

    for i in range(len(p_result) - 1, -1, -1):
        if p_result[i] == None:
            p_result.pop()

    return p_result


# 凸包 ////////////////////////

def filter_convex_hull(results):
    results_convex_hull = []
    data_convex_hull = []
    for result in results:
        x, y, w, h = cv2.boundingRect(result.bbox_list)
        data_convex_hull.append({"x": int(x), "y": int(y), })
    # print(data_convex_hull)
    try:
        result_graham = graham_scan(data_convex_hull)
        # print(result_graham)
        for result in results:
            x, y, w, h = cv2.boundingRect(result.bbox_list)
            for value in result_graham:
                if int(x) == value['x'] and int(y) == value['y']:
                    results_convex_hull.append(result)
    except:
        print('cannot find convex_hull')
        results_convex_hull = results

    return results_convex_hull


if __name__ == '__main__':
    ps = [{"x": 2, "y": 2}, {"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 1.5, "y": 1.5}, {"x": 1, "y": 2},
          {"x": 3, "y": 1.5},
          {"x": 1.5, "y": 1.2}, {"x": 0.5, "y": 2}, {"x": 1.5, "y": 0.5}]
    print(graham_scan(ps))
