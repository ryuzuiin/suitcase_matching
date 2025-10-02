# module/sort_tracker.py


import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def linear_assignment(cost_matrix):
    """
    ハンガリアンアルゴリズムによる線形割り当て問題の解決

    Args:
        cost_matrix (np.ndarray): コスト行列（通常は負のIoU値）
                                 shape: (n_detections, n_trackers)

    Returns:
        np.ndarray: マッチングペアのインデックス配列
                   shape: (n_matches, 2)
                   各行: [detection_idx, tracker_idx]
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(row_ind, col_ind)))

def iou_batch(bb_test, bb_gt):
    """
    複数のバウンディングボックス間のIoU（Intersection over Union）を一括計算

    Args:
        bb_test (np.ndarray): テスト用バウンディングボックス配列
                             shape: (N, 4)、各行: [x1,y1,x2,y2]
        bb_gt (np.ndarray): グラウンドトゥルース（参照）ボックス配列
                          shape: (M, 4)、各行: [x1,y1,x2,y2]

    Returns:
        np.ndarray: IoU行列 shape: (N, M)
                   要素(i,j): bb_test[i]とbb_gt[j]のIoU値（0.0〜1.0）

    計算方法:
        IoU = 交差面積 / 和集合面積
        和集合面積 = ボックス1面積 + ボックス2面積 - 交差面積

    実装詳細:
        - ブロードキャスティングを使用して効率的に計算
        - 交差がない場合はIoU = 0
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    # 交差領域の座標計算
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    # 交差領域の幅と高さ
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    # IoU計算
    wh = w * h  # 交差面積
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    """
    バウンディングボックスを観測ベクトル形式に変換

    Args:
        bbox (array-like): [x1,y1,x2,y2]形式のバウンディングボックス
                          x1,y1: 左上座標、x2,y2: 右下座標

    Returns:
        np.ndarray: [x,y,s,r]形式の観測ベクトル shape: (4, 1)
                   x,y: ボックス中心座標
                   s: スケール（面積）
                   r: アスペクト比（幅/高さ）

    変換理由:
        カルマンフィルターでの追跡に適した表現
        - 中心座標: 動きの追跡が容易
        - 面積とアスペクト比: サイズ変化の独立した追跡
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # 面積
    r = w / float(h)  # アスペクト比
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    状態ベクトルをバウンディングボックス形式に逆変換

    Args:
        x (np.ndarray): [x,y,s,r,...]形式の状態ベクトル
                       x,y: 中心座標、s: 面積、r: アスペクト比
        score (float, optional): 検出信頼度スコア

    Returns:
        np.ndarray: バウンディングボックス
                   scoreなし: shape (1,4) [x1,y1,x2,y2]
                   scoreあり: shape (1,5) [x1,y1,x2,y2,score]

    逆変換式:
        w = sqrt(s * r)  （面積とアスペクト比から幅を計算）
        h = s / w        （面積と幅から高さを計算）
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class KalmanBoxTracker(object):
    """
    カルマンフィルターを使用した個別オブジェクトトラッカー

    このクラスは、単一のオブジェクトの追跡状態を管理し、
    カルマンフィルターによる動き予測と状態更新を行います。

    状態ベクトル（7次元）:
        [x, y, s, r, vx, vy, vs]
        - x, y: ボックス中心座標
        - s: スケール（面積）
        - r: アスペクト比
        - vx, vy: 中心座標の速度
        - vs: スケールの変化率

    観測ベクトル（4次元）:
        [x, y, s, r]（速度は観測できない）
    """
    count = 0  # 全トラッカー共通のIDカウンター

    def __init__(self, bbox):
        """
        初期バウンディングボックスでトラッカーを初期化

        Args:
            bbox (array-like): 初期バウンディングボックス [x1,y1,x2,y2]

        カルマンフィルター設定:
            - F: 状態遷移行列（等速運動モデル）
            - H: 観測行列（位置とサイズのみ観測）
            - R: 観測ノイズ共分散（スケール成分を大きく設定）
            - P: 初期誤差共分散（速度成分を大きく設定）
            - Q: プロセスノイズ共分散
        """
        # 7状態、4観測のカルマンフィルター
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # 状態遷移行列（等速運動モデル）
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])

        # 観測行列
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        # ノイズパラメータの調整
        self.kf.R[2:,2:] *= 10.   # スケール観測ノイズを大きく
        self.kf.P[4:,4:] *= 1000. # 速度の初期不確実性を大きく
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01  # スケール変化率のプロセスノイズを小さく
        self.kf.Q[4:,4:] *= 0.01  # 速度のプロセスノイズを小さく

        # 初期状態設定
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        # トラッキング管理変数
        self.time_since_update = 0  # 最終更新からの経過時間
        self.id = KalmanBoxTracker.count  # ユニークID
        KalmanBoxTracker.count += 1
        self.history = []  # 予測履歴
        self.hits = 0  # 総更新回数
        self.hit_streak = 0  # 連続更新回数
        self.age = 0  # トラッカーの年齢（フレーム数）

    def update(self, bbox):
        """
        観測されたバウンディングボックスで状態を更新

        Args:
            bbox (array-like): 観測されたバウンディングボックス [x1,y1,x2,y2]

        動作:
            - カルマンフィルターの更新ステップ実行
            - トラッキング統計のリセット/更新
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        次フレームの状態を予測

        Returns:
            np.ndarray: 予測されたバウンディングボックス [x1,y1,x2,y2]

        動作:
            - スケール変化の補正（負の面積防止）
            - カルマンフィルターの予測ステップ実行
            - トラッキング統計の更新
        """
        # 負の面積を防ぐ
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if(self.time_since_update > 0):
            self.hit_streak = 0  # 連続更新をリセット
        self.time_since_update += 1

        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        現在の推定バウンディングボックスを取得

        Returns:
            np.ndarray: 現在の状態のバウンディングボックス [x1,y1,x2,y2]
        """
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
    """
    検出結果と既存トラッカーの関連付け

    Args:
        detections (np.ndarray): 検出バウンディングボックス配列
                               shape: (N, 5) [x1,y1,x2,y2,score]
        trackers (np.ndarray): トラッカーのバウンディングボックス配列
                             shape: (M, 5) [x1,y1,x2,y2,0]
        iou_threshold (float): マッチング用IoU閾値（デフォルト: 0.3）

    Returns:
        tuple: (matches, unmatched_detections, unmatched_trackers)
            - matches: マッチしたペア shape: (K, 2)
            - unmatched_detections: 未マッチ検出インデックス
            - unmatched_trackers: 未マッチトラッカーインデックス

    アルゴリズム:
        1. IoU行列の計算
        2. ハンガリアンアルゴリズムで最適割り当て
        3. IoU閾値による割り当ての検証
        4. マッチ/未マッチの分類
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # 完全な1対1マッチング
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # ハンガリアンアルゴリズムで最適化
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    # 未マッチ検出の特定
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)

    # 未マッチトラッカーの特定
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    # IoU閾値によるフィルタリング
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    """
    SORTアルゴリズムのメインクラス

    複数オブジェクトの追跡を管理し、検出結果から
    一貫したIDを持つトラックを生成します。
    """

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        SORTトラッカーの初期化

        Args:
            max_age (int): トラッカーを削除するまでの未更新フレーム数
                         デフォルト: 1（即座に削除）
            min_hits (int): トラックを報告するまでの最小更新回数
                          デフォルト: 3（ノイズ除去）
            iou_threshold (float): データ関連付けのIoU閾値
                                 デフォルト: 0.3

        パラメータの影響:
            - max_age大: オクルージョンに強いが、誤追跡のリスク
            - min_hits大: ノイズに強いが、検出遅延
            - iou_threshold大: 厳密なマッチングだが、追跡失敗のリスク
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []  # アクティブなトラッカーリスト
        self.frame_count = 0  # 処理したフレーム数

    def update(self, dets=np.empty((0, 5))):
        """
        フレームごとの追跡更新

        Args:
            dets (np.ndarray): 検出結果配列 shape: (N, 5)
                             各行: [x1,y1,x2,y2,score]
                             空の場合: (0, 5)の配列

        Returns:
            np.ndarray: 追跡結果配列 shape: (M, 5)
                       各行: [x1,y1,x2,y2,id]
                       idは1から始まる整数

        処理フロー:
            1. 全トラッカーの予測
            2. 検出とトラッカーのマッチング
            3. マッチしたトラッカーの更新
            4. 新規検出に対する新規トラッカー生成
            5. 古いトラッカーの削除
            6. 条件を満たすトラックの返却

        重要:
            空の検出でも毎フレーム呼び出す必要あり（予測のため）
        """
        self.frame_count += 1

        # 既存トラッカーの予測
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # NaN値を含む行を削除
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # データ関連付け
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # マッチしたトラッカーの更新
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 新規トラッカーの生成
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)

        # 結果の収集と古いトラッカーの削除
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            # 報告条件: 最近更新された かつ (十分な更新回数 または 初期フレーム)
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))  # ID+1で1始まり
            i -= 1
            # 古いトラッカーの削除
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))