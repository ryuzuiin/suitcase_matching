# module/frame_processor.py
from ultralytics import YOLO
import numpy as np
from ultralytics.engine.results import Results
from typing import List, Dict, Optional, Tuple

from module.pose_processor import KeypointsToBBoxConverter, ConversionMethod,batch_convert_keypoints
from module.sort_tracker import Sort
from module.draw_utils import draw_tracked_bboxes


class FrameProcessor:
    """
    単一フレームに対する推論 → キーポイント変換 → SORT 追跡 → 可視化までを一括で行う処理クラス。

    本クラスは、YOLO（Pose）モデルを用いたフレーム推論結果（キーポイント）を
    バウンディングボックスへ変換し、SORT トラッカーで ID 付与／更新したのち、
    描画ユーティリティで可視化済みフレームを返します。人物追跡パイプラインの
    1 ステップとして利用できます。

    属性:
        model (YOLO):
            推論に用いる YOLO 互換モデル（例: yolo11m-pose など）。
        bbox_converter (KeypointsToBBoxConverter):
            キーポイント配列を矩形（[x1, y1, x2, y2, conf]）へ変換するコンバータ。
        sort_tracker (Sort):
            追跡用 SORT インスタンス。`max_age`, `min_hits`, `iou_threshold` を内部で保持。

    期待する入力と前提:
        - `model(frame)` は `Results` 配列を返し、`results[0].keypoints.data` が
          `torch.Tensor`（N×K×3 など）を返すことを想定。
        - 変換後の検出は SORT の入力形式 `[x1, y1, x2, y2, score]` を満たすこと。
        - NMS はモデル側または上流で実施されている前提。
    """

    def __init__(self, model: YOLO, sort_max_age: int = 50, sort_min_hits: int = 2, iou_threshold: float = 0.2):
        """
        FrameProcessor を初期化する。

        引数:
            model (YOLO):
                推論に用いる YOLO（Pose）モデル。
            sort_max_age (int):
                SORT の `max_age`。検出が一定フレーム失われてもトラックを保持する猶予（既定 50）。
            sort_min_hits (int):
                SORT の `min_hits`。十分な確信が得られるまでトラックを確定しない最少ヒット数（既定 2）。
            iou_threshold (float):
                SORT の関連付けで用いる IoU しきい値（既定 0.2）。

        備考:
            - これらのパラメータはトラッキングの安定性と再識別の頻度に影響します。
        """
        self.model = model
        self.bbox_converter = KeypointsToBBoxConverter()
        self.sort_tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=iou_threshold)

    

    def process_frame(self, frame: np.ndarray, is_full_process: bool = True,
                    return_data: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray, list]:
        """
        単一フレームを処理して、追跡結果の可視化フレーム（および必要に応じて中間データ）を返す。

        ワークフロー:
            1) YOLO でフレームを推論し、キーポイントを取得
            2) キーポイント → バウンディングボックスへ一括変換（`batch_convert_keypoints`）
            3) SORT に矩形配列 `[x1,y1,x2,y2,score]` を入力してトラック更新
            4) `draw_tracked_bboxes` で ID 付きの矩形を描画して返却
            5) `return_data=True` の場合は、追跡済み矩形と生のキーポイントの一部も併せて返却

        引数:
            frame (np.ndarray):
                入力フレーム（BGR / OpenCV 互換）。形状は (H, W, 3) を想定。
            is_full_process (bool):
                True の場合、推論〜追跡〜描画まで全処理を実行。
                False の場合、処理をスキップして入力フレームをそのまま返す。
            return_data (bool):
                True の場合、以下のタプルを返す:
                    (annotated_frame, tracked_objects, raw_kps_list)
                - annotated_frame (np.ndarray): 可視化済みフレーム
                - tracked_objects (np.ndarray): `[x1, y1, x2, y2, tid]` の追跡配列
                - raw_kps_list (list): YOLO の生キーポイント（追跡件数と同数に切り詰め）

        戻り値:
            np.ndarray | Tuple[np.ndarray, np.ndarray, list]:
                - `return_data=False`（既定）: 可視化済みフレームのみを返す。
                - `return_data=True` : `(annotated_frame, tracked_objects, raw_kps_list)` のタプル。

        返却配列の仕様:
            - `tracked_objects` は `np.float32` の形で `[x1, y1, x2, y2, tid]` を持つ。
              `tid` は整数 ID だが、配列型の都合で float で保持される点に注意。
            - `raw_kps_list` は `results.keypoints.data` の先頭から `len(tracked_objects)` 件のみを使用し、
              トラックとの対応を概ね維持する（厳密な 1:1 対応を保証するものではない）。

        注意:
            - `results.keypoints.data` が `None` の場合は空配列として扱う。
            - 変換に失敗した場合や検出がゼロ件の場合、SORT には空配列を入力する。
            - 可視化は `draw_tracked_bboxes` に委譲しており、ID ごとに色分け表示される。
        """
        tracked_objects = np.empty((0, 5), dtype=np.float32)
        if is_full_process:
            # 1) YOLOモデルでフレームを推論し、キーポイントを取得
            results: Results = self.model(frame, verbose=False)[0]  # type: ignore
            keypoints_data = results.keypoints.data.cpu().numpy() if results.keypoints.data is not None else []

            # 2) キーポイントをバウンディングボックスに一括変換
            sort_detections = batch_convert_keypoints(
                keypoints_list=keypoints_data,
                converter=self.bbox_converter,
                method=ConversionMethod.REGIONAL_PRIORITY,
                confidence_threshold=0.1
            )

            dets_np = sort_detections.astype(np.float32) if (hasattr(sort_detections, "astype") and sort_detections.size) \
                    else np.empty((0, 5), dtype=np.float32)

            # 3) SORTトラッカーでトラックを更新
            tracked_objects = self.sort_tracker.update(dets_np)

            # 4) 追跡結果をフレームに描画
            annotated_frame = draw_tracked_bboxes(frame.copy(), tracked_objects)

            if return_data:
                raw_kps_list = list(keypoints_data)[:len(tracked_objects)]
                return annotated_frame, tracked_objects, raw_kps_list

            return annotated_frame
        else:
            # 処理がスキップされた場合は、元のフレームをそのまま返す
            return frame



        
        
    


