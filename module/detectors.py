# module/detectors.py
from typing import List
import numpy as np

def detect_suitcases(frame, model, confidence_threshold: float = 0.4, class_id: int = 28) -> List[List[float]]:
    """
    スーツケース検出結果を単一フレームから抽出して返す関数。

    本関数は、YOLO 系モデル互換の `model` に対して 1 枚のフレームを推論させ、
    指定クラス（既定では COCO の "suitcase" に対応する 28）に該当する検出のみを
    信頼度しきい値でフィルタして、バウンディングボックスと信頼度を返します。

    引数:
        frame:
            推論対象の単一フレーム。`model` が受け付ける画像形式（例: np.ndarray）を想定。
        model:
            YOLO 系互換モデル。`model(frame, classes=[...], verbose=False)` の呼び出しが可能で、
            各結果に `boxes.xyxy`（[x1, y1, x2, y2]）および `boxes.conf` を持つこと。
        confidence_threshold:
            信頼度の下限。これ未満の検出は除外（既定 0.4）。
        class_id:
            抽出対象のクラス ID（既定 28 = COCO の "suitcase"）。

    戻り値:
        List[List[float]]:
            各検出について `[x1, y1, x2, y2, conf]` の形式で格納したリストの配列。
            検出が存在しない場合は空リスト。

    備考:
        - 返却される座標は元フレームのピクセル座標（左上: x1, y1／右下: x2, y2）。
        - 非極大抑制（NMS）が必要な場合は、モデル側の設定または上位処理で行ってください。
    """

    # 画像フレーム情報を物体検知モデルで検知
    results = model(frame, classes=[class_id], verbose=False)

    # 初期化
    dets: List[List[float]] = []

    # 検知した結果を出力形式に変換する
    for r in results: # 複数検知の際に1つの検知情報ごとに処理をする

        # bboxの情報を取得する。但し存在しない場合は、None
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue

        # bboxの座標（絶対値）を取得
        xyxy = boxes.xyxy.cpu().numpy()        # (N,4)

        # 確信度を取得
        conf = boxes.conf.cpu().numpy().reshape(-1)  # (N,)

        # 
        for i in range(xyxy.shape[0]):

            # 信頼度の取得 XXX
            c = float(conf[i])

            # 閾値による判断
            if c > confidence_threshold:
                x1, y1, x2, y2 = xyxy[i].tolist()
                dets.append([x1, y1, x2, y2, c])

    return dets
