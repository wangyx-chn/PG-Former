import cv2
import numpy as np
from pycocotools import mask as mask_utils
from shapely import Polygon

def approximate_sampling_with_error(sequence, target_points=100):
    """
    加入步长误差校正的均匀近似采样。
    
    :param sequence: 原始输入序列（list 或 NumPy 数组）。
    :param target_points: 希望采样的点数（int）。
    :return: 采样后的序列（list）。
    """
    n = len(sequence)

    # 目标步长（期望步长）
    step = n / target_points
    sampled_indices = []  # 存储采样点的索引

    # 累积误差，用于调整步长
    error = 0
    current = 0  # 从第一个点开始

    for _ in range(target_points):
        sampled_indices.append(int(current))  # 记录当前索引
        current += int(step)  # 按照 step 前进
        error += step - int(step)  # 累积浮点步长的误差

        if error >= 1:
            current += 1  # 如果误差积累到超过 1，就增加步长
            error -= 1  # 重置误差

        # 防止索引超出范围
        if current >= n:
            current = n - 1

    # 根据索引提取采样点
    sampled_sequence = [sequence[int(i)] for i in sampled_indices]
    return sampled_sequence

def rle2polygon(rle):
    '''
    rle转polygon，并实行几何化
    '''
    mask = mask_utils.decode(rle)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)>1:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]
    perimeter = cv2.arcLength(polygon, True)
    epsilon = 0
    while polygon.shape[0]>100:
        epsilon+=1
        polygon = cv2.approxPolyDP(polygon, epsilon, True)
    # # print(polygon.shape)
    # point_thr = 100
    # if len(polygon>point_thr):
    #     polygon = approximate_sampling_with_error(polygon,point_thr)
    polygon = np.array(polygon).reshape(-1, 2)

    return polygon

def match_gt_pred(gt_rles,pred_rles):
    '''
    通过rle的iou计算，匹配gt和pred
    允许多个pred匹配一个gt
    返回的列表包含gt的索引和pred的rle
    '''
    pairs = []
    for pred_rle in pred_rles:
        for i, gt_rle in enumerate(gt_rles):
            if mask_utils.iou([pred_rle],gt_rle,[0])>0.5:
                pairs.append((i,rle2polygon(pred_rle)))
    return pairs

def square_bbox(bbox):
    '''
    把一个bbox转换成最小外接正方形
    '''
    x0, y0, w, h = bbox
    if h>=w:
        return [int(x0-(h-w)/2), y0, h, h]
    else:
        return [x0, int(y0-(w-h)/2), w, w]
    
def ensure_ccw(coords):
    '''
        确保一个[N,2]的polygon坐标序列是顺时针
    '''
    polygon = Polygon(coords)

    # 检查多边形是否是顺时针
    if polygon.exterior.is_ccw:
        # 如果是逆时针，反转顺序
        coords = np.array(polygon.exterior.coords)[::-1][:-1]  # 注意去掉最后的闭合点
    else:
        # 如果已经是顺时针，直接提取坐标
        coords = np.array(polygon.exterior.coords)[:-1]  # 去掉最后的闭合点

    return coords

def sequence_matching_min_cost(A, B):
    """
    保序匹配 A 和 B 序列，确保 B 序列所有点均被 A 中不同点匹配，总代价最小。
    参数：
        A: 序列 A 的点（数组，例如形状为 N x 2）
        B: 序列 B 的点（数组，例如形状为 M x 2）
    返回：
        dp: 动态规划表
        match: 匹配路径
    """
    N = len(A)
    M = len(B)
    
    # 对齐起点
    # start = np.argmin(np.linalg.norm(A-B[0],axis=1))
    # A = np.concatenate([A[start:],A[:start]],axis=0)

    # 动态规划表，初始化为正无穷表示未匹配
    # 创建动态规划表，初始化为正无穷大
    dp = np.full((N + 1, M + 1), float('inf'))
    dp[0][0] = 0  # 起始状态代价为 0
    cost_delta = np.full((N + 1, M + 1), float('inf'))
    cost_delta[0][0] = 0
    
    # 构建动态规划表
    for b in range(1, M + 1):  # 遍历序列 B 的点
        pass
        for a in range(1, N + 1):  # 遍历序列 A 的点
            if b>a:
                dp[a][b]=float('inf')
                cost_delta[a][b] = float('inf')
                continue
            cost = np.linalg.norm(A[a - 1] - B[b - 1])  # 匹配代价
            if b==a:
                dp[a][b] = dp[a - 1][b - 1] + cost
                cost_delta[a][b] = cost
                continue
            
            # 三种状态转移：
            # dp[i][j] = min(
            #     dp[i - 1][j - 1] + cost,  # A[i-1] 匹配 B[j-1]
            #     dp[i - 1][j],             # A[i-1] 留空，尝试后续匹配 B[j]
            #     dp[i][j - 1]              # A[i] 重复匹配 B[j-1]
            # )
            cost_delta[a][b] = min(cost, cost_delta[a-1][b], cost_delta[a-1][b], cost_delta[a-1][b] )
            dp[a][b] = min(dp[a-1][b] - cost_delta[a-1][b],\
                           dp[a][b-1] - cost_delta[a][b-1],\
                             dp[a-1][b-1] - cost_delta[a-1][b-1]) + cost_delta[a][b]
    
    # 回溯匹配路径
    match = []
    i, j = N, M
    while i > 0 and j > 0:
        if cost_delta[i][j] == np.linalg.norm(A[i - 1] - B[j - 1]):
            match.append((i - 1, j - 1))  # A[i-1] 匹配到 B[j-1]
            i -= 1
            j -= 1
        else:
            i -= 1 

    match.reverse()
    return dp, match, A, B