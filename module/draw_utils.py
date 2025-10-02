# module/draw_utils.py
import cv2
import numpy as np
from typing import List, Tuple, Optional

_TRACK_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue (BGR)
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
]

def _draw_box_with_label(frame: np.ndarray, xyxy: List[float], color: Tuple[int,int,int], label: str, font_scale: float=0.7):
    """
    トラッキングされたバウンディングボックスを描画する関数。

    本関数は、人物や物体のトラッキング結果を入力フレームに重ね描きし、
    ID ラベルや（必要に応じて）検出スコアを表示します。トラッキング対象ごとに
    固有の色を割り当て、可視的に区別できるようにします。

    引数:
        frame:
            描画対象の画像フレーム（np.ndarray）。
        tracks:
            トラッキング結果の配列。各要素は `[x1, y1, x2, y2, tid]` 形式。
        dets:
            オプション。検出結果の配列。各要素は `[x1, y1, x2, y2, score]`。
            show_score が True の場合に利用される。
        show_score:
            True の場合、IoU に基づき検出スコアをラベルに併記する。

    戻り値:
        np.ndarray:
            トラッキング結果を可視化したフレーム。
    """
    x1, y1, x2, y2 = map(int, xyxy[:4])
    # バウンディングボックスの矩形を描画
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # ラベルの背景ボックスを描画するためのサイズを計算
    (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color, -1)

    # ラベルテキストを描画
    cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    # バウンディングボックスのサイズ（幅x高さ）を描画
    w, h = x2 - x1, y2 - y1
    cv2.putText(frame, f"{w}x{h}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def _iou_xyxy(a, b) -> float:
    """
    2つの矩形領域の IoU (Intersection over Union) を計算する関数。

    本関数は、矩形を `[x1, y1, x2, y2]` 形式で受け取り、
    交差領域と結合領域の比率を算出します。IoU は物体検出や
    トラッキングにおけるボックスの重なり具合を評価する指標として
    広く利用されます。

    引数:
        a:
            最初の矩形。形式は `[x1, y1, x2, y2]`。
        b:
            2つ目の矩形。形式は `[x1, y1, x2, y2]`。

    戻り値:
        float:
            IoU 値（0.0 ～ 1.0）。
            0.0 は重なりなし、1.0 は完全一致を意味する。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # 交差領域の座標を計算
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    # 交差領域の幅と高さを計算
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    # 各矩形の面積を計算し、IoU を算出
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def draw_tracked_bboxes(frame: np.ndarray, tracks: np.ndarray, *, dets: Optional[np.ndarray]=None, show_score: bool=False) -> np.ndarray:
    """
    トラッキングされたバウンディングボックスをフレームに描画する関数。

    本関数は、トラッキング結果（人物や物体の位置情報）を入力フレームに
    可視化し、各トラックに固有の色と ID ラベルを付与します。また、
    オプションとして検出結果 (dets) を利用し、IoU (交差領域比) に基づいて
    スコアをラベルに併記することが可能です。

    引数:
        frame:
            描画対象の画像フレーム (np.ndarray)。
            OpenCV 互換の BGR カラー画像を想定。
        tracks:
            トラッキング結果の配列 (np.ndarray)。
            各要素は `[x1, y1, x2, y2, tid]` 形式で、矩形の左上座標 (x1,y1)、
            右下座標 (x2,y2)、およびトラック ID (tid) を含む。
        dets:
            検出結果の配列 (np.ndarray)。各要素は `[x1, y1, x2, y2, score]`。
            show_score が True の場合に利用され、IoU が最も高い検出スコアを
            ラベルに併記する。
        show_score:
            True の場合、トラッキング結果と検出結果を IoU で関連付け、
            対応する検出スコアをラベルに表示する。
            False の場合、スコアは表示されない。

    戻り値:
        np.ndarray:
            トラッキング結果を可視化したフレーム。
            各トラックは固有の色で描画され、ラベル (ID または ID＋スコア) が付与される。

    使用例:
        - 行人や荷物の追跡結果を可視化して確認する。
        - トラッキングアルゴリズムの動作をデバッグする際の補助。
    """

    if tracks is None or tracks.size == 0:
        return frame

    use_score = show_score and dets is not None and dets.size > 0
    det_boxes = dets[:, :4] if use_score else None
    det_scores = dets[:, 4] if use_score else None

    for x1, y1, x2, y2, tid in tracks:
        tid = int(tid)
        # IDに基づいて色を決定
        color = _TRACK_COLORS[(tid - 1) % len(_TRACK_COLORS)]

        label = f"ID: {tid}"
        if use_score:
            # トラッキングボックスと最も重なる検出ボックスを探す
            ious = np.array([_iou_xyxy([x1, y1, x2, y2], db) for db in det_boxes])
            j = int(np.argmax(ious)) if ious.size else -1
            if j >= 0 and ious[j] > 0:
                # 重なりがあれば検出スコアをラベルに追加
                label += f" ({float(det_scores[j]):.2f})"

        _draw_box_with_label(frame, [x1, y1, x2, y2], color, label)

    return frame


class ImprovedVideoOverlayRenderer:
    """
    動画フレームに統計情報および拡張トラッキング情報をオーバーレイ描画するクラス。

    本クラスは、リアルタイムのマッチング統計（人物数、荷物数、マッチング率など）や、
    人物とスーツケースのトラッキング結果を可視化し、直感的に状況を把握できる
    インターフェースを提供します。

    主な機能:
        - `draw_statistics_overlay`: 統計情報のオーバーレイボックスを描画し、
          現在の人数、荷物数、マッチング率などを表示。
        - `draw_enhanced_tracking_overlay`: 人物と荷物の追跡結果を描画し、
          マッチ済みペアを線で結び、信頼度を付与。

    利用場面:
        - 駅や空港の監視映像における人物-荷物マッチングのリアルタイム可視化
        - トラッキングアルゴリズムの動作確認・デバッグ
    """

    def __init__(self, frame_width: int, frame_height: int):
        """
        描画器の初期化。

        引数:
            frame_width:
                フレーム幅（ピクセル単位）。
            frame_height:
                フレーム高さ（ピクセル単位）。

        備考:
            - 統計表示ボックスの位置 (右下付近) が自動的に設定される。
            - `stats_x`, `stats_y` は描画基準位置。
        """
        self.w, self.h = frame_width, frame_height
        # 統計表示ボックスの描画位置（右下）を設定
        self.stats_x = self.w - 300
        self.stats_y = self.h - 150

    def draw_statistics_overlay(self, frame: np.ndarray, realtime_statistics):
        """
        リアルタイム統計情報をフレームにオーバーレイ描画する。

        引数:
            frame:
                描画対象の画像フレーム (np.ndarray)。
            realtime_statistics:
                `get_realtime_statistics()` メソッドを持つオブジェクト。
                戻り値は以下の辞書を想定:
                    - 'current_persons': 現在検出された人物数
                    - 'current_suitcases': 現在検出されたスーツケース数
                    - 'persons_with_suitcase': 荷物を持つ人物数
                    - 'persons_without_suitcase': 荷物を持たない人物数
                    - 'unmatched_suitcases': 未マッチのスーツケース数
                    - 'matching_rate': マッチング率 (0〜100%)

        戻り値:
            np.ndarray:
                統計ボックスと数値情報を描画したフレーム。

        特徴:
            - 黒背景＋半透明の統計ボックスを右下に表示。
            - マッチング率に応じたインジケータ (緑・黄・赤) を円で表示。
        """
        stats = realtime_statistics.get_realtime_statistics()
        box_w, box_h = 280, 120
        x0, y0 = self.stats_x - 10, self.stats_y - 10

        # 半透明の背景ボックスをコピーフレームに描画し、元のフレームと合成
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 2)

        # 統計レポートのヘッダーと区切り線を描画
        cv2.putText(frame, "Real-time Matching Stats", (self.stats_x, self.stats_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.line(frame, (self.stats_x, self.stats_y + 30),
                 (self.stats_x + 250, self.stats_y + 30), (255,255,255), 1)

        lines = [
            f"Current: {stats['current_persons']}P, {stats['current_suitcases']}S",
            f"With Bag: {stats['persons_with_suitcase']} ({stats['matching_rate']:.0f}%)",
            f"Without: {stats['persons_without_suitcase']}",
            f"Unmatched Bags: {stats['unmatched_suitcases']}",
        ]
        colors = [(200,200,200),(0,255,0),(0,255,255),(0,0,255)]
        # 各統計行を描画
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (self.stats_x, self.stats_y + 50 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)

        # マッチング率に応じたカラーインジケータ（円）を描画
        rate = stats['matching_rate']
        ind_color = (0,255,0) if rate>75 else ((0,255,255) if rate>50 else (0,0,255))
        cv2.circle(frame, (self.stats_x + 230, self.stats_y + 90), 8, ind_color, -1)
        return frame

    def draw_enhanced_tracking_overlay(self, frame: np.ndarray,
                                       person_tracks: np.ndarray,
                                       suitcase_tracks: np.ndarray,
                                       confirmed_matches: dict,
                                       matcher) -> np.ndarray:
        
        """
        人物とスーツケースのトラッキング情報を拡張表示する。

        引数:
            frame:
                描画対象のフレーム (np.ndarray)。
            person_tracks:
                人物のトラッキング結果。各要素は `[x1, y1, x2, y2, pid]`。
            suitcase_tracks:
                スーツケースのトラッキング結果。各要素は `[x1, y1, x2, y2, sid]`。
            confirmed_matches:
                人物 ID とスーツケース ID の対応辞書 `{pid: sid}`。
            matcher:
                マッチング信頼度を返すオブジェクト。
                `matcher.get_match_confidence(pid)` が利用可能であること。

        戻り値:
            np.ndarray:
                人物・スーツケースのバウンディングボックスとラベルを描画し、
                マッチ済みの組み合わせを線で結んだフレーム。

        特徴:
            - マッチ済み人物は緑、未マッチ人物は灰色で描画。
            - マッチ済みスーツケースはオレンジ、未マッチは灰色。
            - 人物とスーツケース間を線で結び、対応関係を直感的に表示。
            - ラベルにマッチ信頼度を併記可能。
        """
        matched_pids = set(confirmed_matches.keys())
        # マッチ済み人物のバウンディングボックスを描画
        for x1, y1, x2, y2, tid in person_tracks:
            tid = int(tid)
            ok = tid in matched_pids
            color = (0,255,0) if ok else (128,128,128)
            label = f"{'' if ok else ''}Person-{tid}"
            if ok:
                # マッチングが確定している場合、信頼度スコアをラベルに追加
                label += f" ({matcher.get_match_confidence(tid):.2f})"
            _draw_box_with_label(frame, [x1,y1,x2,y2], color, label)

        # マッチ済みスーツケースのバウンディングボックスを描画
        matched_sids = set(confirmed_matches.values())
        for x1, y1, x2, y2, tid in suitcase_tracks:
            tid = int(tid)
            ok = tid in matched_sids
            s_color = (255,165,0) if ok else (100,100,100)
            _draw_box_with_label(frame, [x1,y1,x2,y2], s_color,
                                 f"{'' if ok else ''}Suitcase-{tid}", font_scale=0.6)

       # マッチングされた人物とスーツケースを線で結ぶ
        for pid, sid in confirmed_matches.items():
            pc = self._center_of(person_tracks, pid)
            sc = self._center_of(suitcase_tracks, sid)
            if pc and sc:
                cv2.line(frame, pc, sc, (0,255,0), 2)
        return frame

    def _center_of(self, tracks: np.ndarray, target_id: int) -> Optional[Tuple[int,int]]:
        """
        指定した ID を持つトラックの中心座標を計算して返す。

        引数:
            tracks:
                トラッキング結果の配列。各要素は `[x1, y1, x2, y2, tid]`。
            target_id:
                中心座標を求めたいトラックの ID。

        戻り値:
            (x, y):
                中心座標 (整数ピクセル)。
            None:
                指定 ID が存在しない場合。
        """
        for x1, y1, x2, y2, tid in tracks:
            if int(tid) == int(target_id):
                return (int((x1+x2)/2), int((y1+y2)/2))
        return None
