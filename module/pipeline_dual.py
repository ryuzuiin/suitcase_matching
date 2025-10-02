# module/pipeline_dual.py
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
from ultralytics import YOLO

from module.sort_tracker import Sort
from module.detectors import detect_suitcases
from module.pose_processor import KeypointsToBBoxConverter, ConversionMethod, batch_convert_keypoints
from module.matching import MatcherBundle
from module.draw_utils import ImprovedVideoOverlayRenderer


class DualPipeline:
    """
    デュアルパイプライン（人 + 荷物）。

    このクラスは、人物と荷物（バックパック、ハンドバッグ、スーツケース）を同時に
    検出・追跡し、それらの関連付けを行うためのエンドツーエンドの処理パイプラインです。

    ### 処理フロー
    本パイプラインはフレームごとに以下の主要なステップを実行します。

    1. **人物の検出と追跡**:
       - YOLOモデルで人物の姿勢（pose）を検出し、キーポイントからバウンディングボックスを生成します。
       - 生成されたバウンディングボックスをSORTアルゴリズムで追跡します。

    2. **荷物の検出と追跡**:
       - 別途YOLOモデルで荷物クラス（COCO ID: 24, 26, 28）を検出し、それぞれ独立したSORTトラッカーで追跡します。

    3. **データ統合とマッチング**:
       - 人物と荷物の追跡結果を統合し、`MatcherBundle` を用いて人-物間の空間的・時間的なマッチングを行います。

    4. **結果の描画と統計**:
       - 追跡・マッチング結果をフレーム上に描画し、内部で各種統計情報を収集します。
    """

    def __init__(
        self,
        pose_model: YOLO,
        suitcase_model: YOLO,
        frame_width: int,
        frame_height: int,
        conversion_method: ConversionMethod = ConversionMethod.REGIONAL_PRIORITY,
        conversion_kwargs: Dict[str, Any] = None,
        # 荷物検出の信頼度
        suitcase_confidence: float = 0.4,
        backpack_confidence: Optional[float] = None,
        handbag_confidence: Optional[float] = None,
        # COCO クラスID
        backpack_class_id: int = 24,
        handbag_class_id: int = 26,
        suitcase_class_id: int = 28,
        # SORT パラメータ
        person_sort=(30, 1, 0.3),
        bag_sort=(20, 1, 0.2),
        # マッチング条件
        match_dist=250, match_overlap=0.1, match_min_frames=1,
    ):
        """
        DualPipelineの初期化。

        人-物追跡に必要なモデル、トラッカー、コンバータ、および各種パラメータを設定します。

        Args:
            pose_model (YOLO): 人物姿勢検出用のYOLOモデルインスタンス。
            suitcase_model (YOLO): 荷物検出用のYOLOモデルインスタンス。
            frame_width (int): 処理するフレームの幅（ピクセル）。
            frame_height (int): 処理するフレームの高さ（ピクセル）。
            conversion_method (ConversionMethod, optional): キーポイントからBBoxへの変換手法。
                                                           デフォルトは `REGIONAL_PRIORITY`。
            conversion_kwargs (Dict[str, Any], optional): 変換手法に渡す追加パラメータ。
            suitcase_confidence (float, optional): スーツケースの検出信頼度しきい値。
            backpack_confidence (Optional[float], optional): バックパックの検出信頼度しきい値。
                                                               Noneの場合、`suitcase_confidence` を使用。
            handbag_confidence (Optional[float], optional): ハンドバッグの検出信頼度しきい値。
                                                               Noneの場合、`suitcase_confidence` を使用。
            backpack_class_id (int, optional): バックパックのCOCOクラスID。デフォルトは24。
            handbag_class_id (int, optional): ハンドバッグのCOCOクラスID。デフォルトは26。
            suitcase_class_id (int, optional): スーツケースのCOCOクラスID。デフォルトは28。
            person_sort (Tuple, optional): 人物用SORTトラッカーのパラメータ。` (max_age, min_hits, iou_threshold)`。
            bag_sort (Tuple, optional): 荷物用SORTトラッカーのパラメータ。` (max_age, min_hits, iou_threshold)`。
            match_dist (int, optional): 人と荷物のマッチングにおける距離しきい値。
            match_overlap (float, optional): 人と荷物のマッチングにおける重なりしきい値。
            match_min_frames (int, optional): マッチングを確定するために必要な最小一致フレーム数。
        """
        
        self.pose_model = pose_model
        self.suitcase_model = suitcase_model

        self.conv_method = conversion_method
        self.conv_kw = conversion_kwargs or {}

        # 荷物検出の信頼度（指定がなければ suitcase_conf に統一）
        self.suitcase_conf = float(suitcase_confidence)
        self.backpack_conf = float(backpack_confidence) if backpack_confidence is not None else self.suitcase_conf
        self.handbag_conf  = float(handbag_confidence)  if handbag_confidence  is not None else self.suitcase_conf

        # クラスID
        self.backpack_cls = int(backpack_class_id)
        self.handbag_cls  = int(handbag_class_id)
        self.suitcase_cls = int(suitcase_class_id)

        # トラッカー
        self.person_tracker   = Sort(max_age=person_sort[0], min_hits=person_sort[1], iou_threshold=person_sort[2])
        self.backpack_tracker = Sort(max_age=bag_sort[0], min_hits=bag_sort[1], iou_threshold=bag_sort[2])
        self.handbag_tracker  = Sort(max_age=bag_sort[0], min_hits=bag_sort[1], iou_threshold=bag_sort[2])
        self.suitcase_tracker = Sort(max_age=bag_sort[0], min_hits=bag_sort[1], iou_threshold=bag_sort[2])

        # コンバータ / マッチャー / 描画器
        self.converter = KeypointsToBBoxConverter()
        self.bundle = MatcherBundle(
            frame_width=frame_width, frame_height=frame_height,
            distance_threshold=match_dist, overlap_threshold=match_overlap, min_match_frames=match_min_frames
        )
        self.overlay = ImprovedVideoOverlayRenderer(frame_width, frame_height)

        # 統計用カウンタ
        self.bbox_attempt_total = 0
        self.bbox_success_total = 0
        self.trk_stats = {
            'person':   {'total_detections': 0, 'current_active_tracks': 0, 'max_concurrent_tracks': 0},
            'backpack': {'total_detections': 0, 'current_active_tracks': 0, 'max_concurrent_tracks': 0},
            'handbag':  {'total_detections': 0, 'current_active_tracks': 0, 'max_concurrent_tracks': 0},
            'suitcase': {'total_detections': 0, 'current_active_tracks': 0, 'max_concurrent_tracks': 0},
        }

        self.frame_width = frame_width
        self.frame_height = frame_height

    def _update_person_side(self, frame: np.ndarray):
        """
        単一フレームにおける人物の検出と追跡。

        YOLOモデルで人物の姿勢を検出し、キーポイントからバウンディングボックスを生成します。
        生成されたバウンディングボックスは、SORTトラッカーに渡され、追跡状態が更新されます。
        同時に、バウンディングボックス変換の成功率を記録します。

        Args:
            frame (np.ndarray): 処理対象のビデオフレーム。

        Returns:
            Tuple[np.ndarray, list]:
                - `person_tracks`: 追跡結果のNumPy配列。形状は (N, 5) で、各行は [x1, y1, x2, y2, track_id] を表します。
                                   検出がない場合は空の配列を返します。
                - `raw_kps_list`: 各追跡人物に対応する元のキーポイントデータ（リスト）。
                                  追跡結果と順序が対応します。

        Raises:
            AttributeError: YOLO検出結果が予期せぬ形式で、`keypoints`属性を持たない場合。
        """
        person_tracks = np.empty((0, 5), dtype=np.float32)
        raw_kps_list = []

        # 1. Poseモデルによる推論実行 (人物のキーポイントを検出)
        pres = self.pose_model(frame, verbose=False)

        # 2. キーポイントデータの抽出とBBoxへの変換
        if pres and pres[0].keypoints is not None:
            # 生のキーポイントデータ (N,17,3) を抽出
            kps = pres[0].keypoints.data.cpu().numpy()  # (N,17,3)
            if len(kps) > 0:
                # キーポイントからSORT形式のバウンディングボックスを生成
                dets = batch_convert_keypoints(
                    keypoints_list=kps,
                    converter=self.converter,
                    method=self.conv_method,
                    **self.conv_kw
                )
                # BBox変換の試行回数と成功数を統計に記録
                self.bbox_attempt_total += len(kps)
                if dets is not None:
                    self.bbox_success_total += len(dets)

                # 3. 生成されたBBoxを人物用SORTトラッカーに入力し、トラックIDを更新
                person_tracks = self.person_tracker.update(
                    dets.astype(np.float32) if dets is not None else np.empty((0, 5), dtype=np.float32)
                )
                # 今は順序対応で keypoints を切り出す
                raw_kps_list = [kp for kp in kps][:len(person_tracks)]
        else:
            # 検出がない場合、空の入力でトラッカーを更新（トラックの寿命を減らす）
            person_tracks = self.person_tracker.update(np.empty((0, 5), dtype=np.float32))

        # 統計更新
        self.trk_stats['person']['current_active_tracks'] = int(len(person_tracks))
        self.trk_stats['person']['max_concurrent_tracks'] = max(
            self.trk_stats['person']['max_concurrent_tracks'], int(len(person_tracks))
        )
        self.trk_stats['person']['total_detections'] += int(len(raw_kps_list))

        return person_tracks, raw_kps_list

    def _detect_and_track_bag_class(self, frame: np.ndarray, conf: float, cls_id: int, tracker: Sort, stats_key: str):
        """
        単一の荷物クラスに対する検出、追跡、および統計更新。

        指定された信頼度とクラスIDに基づき、荷物を検出して対応するSORTトラッカーを更新します。

        Args:
            frame (np.ndarray): 処理対象のビデオフレーム。
            conf (float): 検出信頼度しきい値。
            cls_id (int): 検出対象のCOCOクラスID。
            tracker (Sort): 使用するSORTトラッカーインスタンス。
            stats_key (str): 統計情報を更新するための辞書キー（例: 'backpack', 'handbag'）。

        Returns:
            np.ndarray: 当該クラスの追跡結果。形状は (N, 5) で、各行は [x1, y1, x2, y2, track_id] を表します。

        Raises:
            KeyError: `stats_key`が`self.trk_stats`辞書に存在しない場合。
        """
        # 1. 荷物モデルによる検出（指定されたクラスIDと信頼度を使用）
        dets = detect_suitcases(frame, self.suitcase_model, conf, cls_id)

        # 2. 検出結果をNumPy配列に変換し、SORTに入力
        dets_np = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)
        tracks = tracker.update(dets_np)

        # 3. 追跡統計を更新
        self.trk_stats[stats_key]['total_detections'] += int(len(dets) if dets else 0)
        self.trk_stats[stats_key]['current_active_tracks'] = int(len(tracks))
        self.trk_stats[stats_key]['max_concurrent_tracks'] = max(
            self.trk_stats[stats_key]['max_concurrent_tracks'], int(len(tracks))
        )
        return tracks

    def process(self, frame: np.ndarray, frame_no: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        1フレームの全処理を実行するメイン関数。

        人物と3つの荷物クラスを個別に追跡し、それらの結果を統合してマッチングを行います。
        最終的な結果は描画されたフレームとマッチング情報として返されます。

        Args:
            frame (np.ndarray): 処理対象のフレーム。
            frame_no (int): 現在のフレーム番号。

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - `annotated`: 追跡結果が描画されたフレーム。
                - `res`: マッチング結果、`confirmed`（確定ペア）を含む辞書。
        """
        annotated = frame.copy()

        # --- 人物 ---
        person_tracks, raw_kps_list = self._update_person_side(annotated)

        # --- 荷物（3クラス） ---
        bp_tracks = self._detect_and_track_bag_class(
            annotated, self.backpack_conf, self.backpack_cls, self.backpack_tracker, 'backpack'
        )
        hb_tracks = self._detect_and_track_bag_class(
            annotated, self.handbag_conf, self.handbag_cls, self.handbag_tracker, 'handbag'
        )
        sc_tracks = self._detect_and_track_bag_class(
            annotated, self.suitcase_conf, self.suitcase_cls, self.suitcase_tracker, 'suitcase'
        )

        # 荷物トラックを統合し、アイテムタイプ情報（マッチング処理に必要）を生成
        item_types = []
        if len(bp_tracks) > 0:
            item_types.extend(['backpack'] * len(bp_tracks))
        if len(hb_tracks) > 0:
            item_types.extend(['handbag'] * len(hb_tracks))
        if len(sc_tracks) > 0:
            item_types.extend(['suitcase'] * len(sc_tracks))

        # --- bag_tracks として統合 ---
        if (len(bp_tracks) + len(hb_tracks) + len(sc_tracks)) > 0:
            bag_tracks = np.vstack([bp_tracks, hb_tracks, sc_tracks]).astype(np.float32)
        else:
            bag_tracks = np.empty((0, 5), dtype=np.float32)

        # --- マッチング ---
        res = self.bundle.step(frame_no, person_tracks, bag_tracks, raw_kps_list, item_types)
        confirmed = res["confirmed"]

        # --- 描画 ---
        annotated = self.overlay.draw_enhanced_tracking_overlay(
            annotated, person_tracks, bag_tracks, confirmed, self.bundle.matcher
        )
        annotated = self.overlay.draw_statistics_overlay(annotated, self.bundle.rt_stats)

        return annotated, res

    def finalize(self) -> Dict[str, Any]:
        """
        パイプラインの終了処理。

        すべてのフレーム処理が完了した後に呼び出され、最終的な統計情報を計算し、辞書形式で返します。
        これにより、パイプライン全体のパフォーマンスを評価できます。

        Returns:
            Dict[str, Any]: 追跡、マッチング、バウンディングボックス変換など、すべての統計情報を含む辞書。
        """
        rt = self.bundle.finalize()
        # BBox変換の最終成功率を計算
        overall_bbox_rate = (self.bbox_success_total / self.bbox_attempt_total * 100.0) if self.bbox_attempt_total else 0.0

        # 最終レポートにすべての追跡統計とBBox変換率を追加
        rt.update({
            'person_tracking':   self.trk_stats['person'],
            'backpack_tracking': self.trk_stats['backpack'],
            'handbag_tracking':  self.trk_stats['handbag'],
            'suitcase_tracking': self.trk_stats['suitcase'],
            'bbox_stats': {
                'total_attempts': self.bbox_attempt_total,
                'successful_conversions': self.bbox_success_total,
                'success_rate': overall_bbox_rate,
            }
        })
        return rt