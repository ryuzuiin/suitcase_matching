# module/matching.py

from typing import Dict, Any, List, Optional
import numpy as np
import math

class TrackingDataCollector:
    """
    追跡データ収集器。

    このクラスは、ビデオの各フレームにおける人物と荷物の追跡データを構造化された
    形式で収集・保存します。これにより、後続のモジュールがデータを扱いやすくなります。

    ### 実装内容
    フレームごとの追跡結果（NumPy配列）を、追跡ID、バウンディングボックス、
    中心座標、面積などの情報を含むPython辞書に変換します。
    この辞書は、人物と荷物それぞれのリストとして一つのフレーム情報辞書にまとめられます。
    また、過去Nフレーム分のデータを保持し、リアルタイムでのマッチング処理に利用されます。
    
    """

    def __init__(self, max_frames_history=30):
        """
        トラッカーデータコレクタの初期化。

        Args:
            max_frames_history (int, optional): 履歴として保持するフレームの最大数。
                                                 デフォルトは30フレーム。
        """
        self.frame_data = []
        self.max_frames_history = max_frames_history
        self.current_frame_data = None

    def add_frame_data(self, frame_number, person_tracks, suitcase_tracks, raw_keypoints=None, item_types=None):
        """
        単一フレームの追跡データを構造化して追加します。

        このメソッドは、SORTトラッカーから出力された生の追跡結果（NumPy配列）を、
        人間が読みやすく、かつ後続のモジュールが扱いやすい辞書形式に変換します。

        ### 処理フロー
        1.  `frame_info` 辞書を初期化し、フレーム番号を格納します。
        2.  `person_tracks` 配列をループ処理し、各人物のトラッキング情報（ID、BBox、中心座標など）を抽出して辞書に格納します。
        3.  キーポイントデータが利用可能であれば、対応する人物データ辞書に追加します。
        4.  同様に、`suitcase_tracks` 配列をループし、荷物のトラッキング情報を抽出して辞書に格納します。
        5.  完成した `frame_info` 辞書を、履歴リスト `self.frame_data` に追加します。
        6.  履歴の長さが `max_frames_history` を超えないように、最も古いデータを削除します。

        Args:
            frame_number (int): 現在のフレーム番号。
            person_tracks (np.ndarray): 人物追跡結果。形状は (N, 5) で、各行は [x1, y1, x2, y2, track_id] を表します。
            suitcase_tracks (np.ndarray): 荷物追跡結果。形状は (M, 5) で、各行は [x1, y1, x2, y2, track_id] を表します。
            raw_keypoints (np.ndarray, optional): 人物追跡結果に対応する元のキーポイント配列。
                                                  形状は (N, 17, 3)。
            item_types (List[str], optional): 荷物追跡結果に対応する物品タイプ（"backpack", "handbag"など）のリスト。

        Raises:
            IndexError: `suitcase_tracks`と`item_types`の長さが一致しない場合、タイプ情報へのアクセスで発生する可能性があります。
        """
        item_types = item_types or []  # デフォルト値の追加処理
        
        frame_info = {
            'frame': frame_number,
            'persons': [],
            'suitcases': []
        }

        # 人物データを処理
        for i, track in enumerate(person_tracks):
            x1, y1, x2, y2, track_id = track.astype(int)
            person_data = {
                'id': int(track_id),
                'bbox': [x1, y1, x2, y2],
                'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                'area': (x2 - x1) * (y2 - y1)
            }

            #  直接対応：tracks[i] ↔ raw_keypoints[i]
            if raw_keypoints is not None and i < len(raw_keypoints):
                person_data['keypoints'] = raw_keypoints[i]
            else:
                person_data['keypoints'] = None

            frame_info['persons'].append(person_data)

        # スーツケースデータを処理 - この部分を修正
        for i, track in enumerate(suitcase_tracks):
            x1, y1, x2, y2, track_id = track.astype(int)
            suitcase_data = {
                'id': int(track_id),
                'bbox': [x1, y1, x2, y2],
                'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                'area': (x2 - x1) * (y2 - y1),
                'type': item_types[i] if i < len(item_types) else 'suitcase'  # タイプ情報を追加
            }
            frame_info['suitcases'].append(suitcase_data)

        self.frame_data.append(frame_info)
        self.current_frame_data = frame_info

        if len(self.frame_data) > self.max_frames_history:
            self.frame_data.pop(0)

    def get_current_frame_data(self):
        """
        直近に追加されたフレームの構造化されたデータを取得します。

        このメソッドは、`add_frame_data` メソッドによって最後に処理された
        フレームの情報辞書を返します。

        Returns:
            Dict[str, Any] or None: 最新のフレーム情報。
                                    まだデータが追加されていない場合は`None`を返します。
        """
        return self.current_frame_data


class ImprovedRealtimePersonSuitcaseMatcher:
    # このクラスは、リアルタイムのビデオストリーム内で人物とスーツケースを関連付けるための
    # 強化されたルールベースのアルゴリズムを提供します。
    #
    # 主な機能は以下の通りです。
    # - 距離、位置、姿勢、移動パターンに基づいた多要素のマッチング評価。
    # - マッチングの信頼性を高めるための、動的なスコアリングとフィルタリング。
    # - IDの切り替わりや一時的な遮蔽に対応するための、履歴と所有権の管理。
    # - リアルタイム環境でのパフォーマンスを考慮した設計

    def __init__(self, distance_threshold=200, overlap_threshold=0.1, min_match_frames=1,
                 frame_width=1920, frame_height=1080):
        self.distance_threshold = distance_threshold
        self.overlap_threshold = overlap_threshold
        self.min_match_frames = min_match_frames
        self.match_history = {}
        self.confirmed_matches = {}

        self.first_approach_history = {}
        self.approach_timestamps = {}

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.gate_filter_stats = {
            'position_filtered': 0,
            'area_filtered': 0,
            'repeat_filtered': 0,
            'total_attempts': 0
        }
        self.suitcase_match_history = {}

        # 速度/静止の追跡
        self.prev_person_center = {}
        self.prev_suitcase_center = {}
        self.person_static_streak = {}
        self.suitcase_static_streak = {}
        self.VEL_STATIC_TH = 1.0
        self.STATIC_STREAK_REQ = 8

        # 所有権ロック（強化版）
        self.suitcase_owner = {}
        self.suitcase_owner_ttl = {}
        self.OWNER_TTL = 60
        self.OWNER_BONUS = 0.8
        
        # 位置メモリとID変化の処理
        self.position_memory = {}
        self.current_frame_count = 0
        self.POSITION_TOLERANCE = 80
        self.MEMORY_FRAMES = 45
        
        # 身長と年齢の推定
        self.ADULT_HEIGHT_RATIO = 0.25
        self.CHILD_HEIGHT_RATIO = 0.18
        self.ADULT_REQUIRED_WHEN_SUITCASE_STATIC = True
        self.STRONG_POSTURE_DIST = 100
        
        # 動的距離しきい値
        self.BASE_DISTANCE_THRESHOLD = 250
        self.TALL_PERSON_BONUS = 50

    def _quantize_position(self, center):
        """位置の量子化（位置メモリ用）

        入力された座標を、設定された許容範囲（POSITION_TOLERANCE）に基づき、
        離散的なグリッド座標に変換します。

        これにより、トラッカーのわずかなブレや振動による座標の変動を吸収し、
        安定した相対位置関係のキーを生成します。

        Args:
            center (tuple[float, float]): 物体（人物またはスーツケース）の中心座標 (x, y)。

        Returns:
            tuple[int, int]: 量子化されたグリッド座標。
        """
        return (int(center[0] // self.POSITION_TOLERANCE), 
                int(center[1] // self.POSITION_TOLERANCE))  # 座標を許容誤差で割ってグリッド座標を計算

    def _update_position_memory(self, person, suitcase, match_confirmed=False):
        """位置メモリの更新

        `_quantize_position`によって生成された人物とスーツケースの相対位置情報を、
        位置メモリ辞書に記録または更新します。

        - `match_confirmed`がTrueの場合、現在の相対位置を「成功した組み合わせ」として記憶します。
        - 記憶された情報には、一致したID、最終確認フレーム数、およびマッチング強度が含まれます。
        - このメモリは、IDの切り替わりや一時的な遮蔽が発生した際に、
          過去の成功経験に基づいてマッチングを再確立するために使用されます。

        Args:
            person (dict): 検出された人物のデータ（'id'、'center'を含む）。
            suitcase (dict): 検出されたスーツケースのデータ（'id'、'center'を含む）。
            match_confirmed (bool, optional): マッチングが成功したかどうかのフラグ。
                                             Defaults to False。
        """
        p_pos = self._quantize_position(person['center'])
        s_pos = self._quantize_position(suitcase['center'])
        pos_key = (p_pos, s_pos)  # 人物とスーツケースの量子化された相対位置をキーとする
        
        if match_confirmed:
            self.position_memory[pos_key] = {
                'person_id': person['id'],
                'suitcase_id': suitcase['id'],
                'last_seen': self.current_frame_count,  # 最終確認フレーム数を記録
                'match_strength': self.position_memory.get(pos_key, {}).get('match_strength', 0) + 1  # マッチング強度をインクリメント
            }

    def _get_position_memory_bonus(self, person, suitcase):
        """位置メモリに基づくボーナス計算

        現在の人物とスーツケースの相対位置が、過去に成功した組み合わせと一致するかを
        位置メモリから検索します。

        - 一致するデータが見つかった場合、最終確認からの経過時間とマッチング強度に応じて
          ボーナススコアを計算して返します。
        - 時間が経つにつれてボーナスは減衰し、過去の経験が現在の判断に与える影響を調整します。
        - このボーナススコアは、メインのマッチングスコアに加算され、
          マッチングの信頼性を高めるために使用されます。

        Args:
            person (dict): 検出された人物のデータ（'id'、'center'を含む）。
            suitcase (dict): 検出されたスーツケースのデータ（'id'、'center'を含む）。

        Returns:
            float: 0.0から1.0までのボーナススコア。一致するメモリがない場合は0.0を返します。
        """
        p_pos = self._quantize_position(person['center'])
        s_pos = self._quantize_position(suitcase['center'])
        pos_key = (p_pos, s_pos)
        
        if pos_key in self.position_memory:
            memory = self.position_memory[pos_key]
            frames_ago = self.current_frame_count - memory['last_seen']  # 最終確認からの経過フレーム数
            
            
            if frames_ago <= self.MEMORY_FRAMES:
                time_decay = max(0, 1.0 - frames_ago / self.MEMORY_FRAMES)  # 時間減衰係数を計算
                strength_bonus = min(memory.get('match_strength', 1) * 0.1, 0.5)  # 強度に応じたボーナスを計算
                return time_decay * 0.6 + strength_bonus  # 時間減衰と強度ボーナスを合成
        
        return 0.0

    def _estimate_person_height_category(self, person):
        """人物の身長カテゴリを推定

        人物のバウンディングボックスの高さが、フレーム全体の高さに対して占める
        割合を計算し、その比率に基づいて人物の身長カテゴリを推定します。

        この関数は、画面上の人物の大きさを基に、カメラからの相対的な距離を
        間接的に判断するために利用されます。

        Args:
            person (dict): 検出された人物のデータ。`bbox`キー（[x1, y1, x2, y2]）を含む必要があります。

        Returns:
            str: 推定された身長カテゴリ。'adult', 'child', または 'teen' のいずれか。
        """
        height_ratio = (person['bbox'][3] - person['bbox'][1]) / self.frame_height  # BBoxの高さ比率を計算
        
        if height_ratio >= self.ADULT_HEIGHT_RATIO:
            return 'adult'
        elif height_ratio <= self.CHILD_HEIGHT_RATIO:
            return 'child'
        else:
            return 'teen'

    def _calculate_dynamic_distance_threshold(self, person):
        """身長に応じた動的距離しきい値を計算

        `_estimate_person_height_category`関数で推定された人物の身長カテゴリに基づき、
        人物とスーツケース間のマッチングに最適な距離しきい値を動的に決定します。

        - 'adult'の場合: 背の高い人物やカメラから遠い人物でもマッチングできるよう、
          基本しきい値にボーナスを加算します。
        - 'child'の場合: 子供とスーツケースの相対距離は大人よりも狭いという前提に基づき、
          基本しきい値から値を減算します。
        - 'teen'の場合: 基本しきい値をそのまま使用します。

        この動的な調整により、様々な体格の人物に対するマッチング精度を向上させます。

        Args:
            person (dict): 検出された人物のデータ。身長カテゴリの推定に利用されます。

        Returns:
            float: 人物とスーツケースを関連付けるために使用される動的な距離しきい値（ピクセル単位）。
        """
        height_category = self._estimate_person_height_category(person)
        
        if height_category == 'adult':
            return self.BASE_DISTANCE_THRESHOLD + self.TALL_PERSON_BONUS  # 大人/遠い人物向けに閾値を緩和
        elif height_category == 'child':
            return self.BASE_DISTANCE_THRESHOLD - 50  # 子供向けに閾値を厳しくする
        else:
            return self.BASE_DISTANCE_THRESHOLD  # 基本閾値を使用


    def _cleanup_position_memory(self):
        """期限切れの位置メモリをクリーンアップ

        最終確認から一定期間（45フレーム、約4秒）以上経過した人物と
        スーツケースの組み合わせに関する位置メモリを削除します。

        これにより、メモリの使用量を最適化し、古く関連性の低い情報が
        マッチングの判断に影響を与えるのを防ぎます。

        Args:
            self (object): インスタンス自身。`current_frame_count`と`position_memory`、
                        `MEMORY_FRAMES`属性にアクセスするために必要です。
        """
        expired_keys = []
        for pos_key, memory in self.position_memory.items():
            if self.current_frame_count - memory['last_seen'] > self.MEMORY_FRAMES:
                expired_keys.append(pos_key)  # 期限切れのキーをリストアップ
        
        for key in expired_keys:
            del self.position_memory[key]  # 期限切れのメモリを削除

    def calculate_distance(self, person_center, suitcase_center):
        """2点間のユークリッド距離を計算

        人物の中心座標とスーツケースの中心座標の間の直線距離を計算します。
        これは、2つの物体がどれだけ離れているかを測るための基本的な関数です。

        Args:
            person_center (tuple[float, float]): 人物の中心座標 (x, y)。
            suitcase_center (tuple[float, float]): スーツケースの中心座標 (x, y)。

        Returns:
            float: 2点間のユークリッド距離。
        """
        dx = person_center[0] - suitcase_center[0]
        dy = person_center[1] - suitcase_center[1]
        return (dx ** 2 + dy ** 2) ** 0.5  # ユークリッド距離の計算

    def is_in_gate_area(self, suitcase_bbox):
        """スーツケースがゲートエリア内にあるかを判断

        スーツケースのバウンディングボックスの位置、サイズ、および縦横比が、
        事前に設定された厳しいルールをすべて満たすかどうかをチェックします。
        これらの条件は、特定のカメラアングルと設置環境に合わせてハードコードされています。
        すべての条件が満たされた場合にのみ、`True`を返します。

        Args:
            suitcase_bbox (tuple[float, float, float, float]): スーツケースのバウンディングボックス座標 (x1, y1, x2, y2)。

        Returns:
            bool: すべての条件を満たし、スーツケースがゲートエリア内にあると判断されれば`True`、そうでなければ`False`。
        """
        x1, y1, x2, y2 = suitcase_bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        is_upper_area = center_y < self.frame_height * 0.5  # フレーム上半分
        is_right_area = center_x > self.frame_width * 0.65  # フレーム右側
        aspect_ratio = height / width if width > 0 else 0
        is_vertical = aspect_ratio > 1.2   # 縦長である
        is_reasonable_size = (width < self.frame_width * 0.15 and height < self.frame_height * 0.4)  # 適度なサイズ
        is_gate_area = is_upper_area and is_right_area and is_vertical and is_reasonable_size  # すべての条件を満たす

        if is_gate_area:
            print(f"     ゲート検知: 位置({center_x:.0f},{center_y:.0f}) "
                  f"尺寸({width:.0f}x{height:.0f}) 縦横比:{aspect_ratio:.2f}")

        return is_gate_area

    def is_position_relationship_valid(self, person_center, suitcase_center, person_bbox):
        """人物とスーツケースの位置関係が妥当であるかを検証

        人物とスーツケースの相対的な位置関係が、事前に設定されたルールに
        適合するかをチェックします。スーツケースが人物の右斜め上にある、
        または人物に対して高すぎる、不自然な角度にあるといった異常な状況を
        検出します。いずれかの条件に違反した場合、`False`を即座に返します。

        Args:
            person_center (tuple[float, float]): 人物の中心座標 (x, y)。
            suitcase_center (tuple[float, float]): スーツケースの中心座標 (x, y)。
            person_bbox (tuple[float, float, float, float]): 人物のバウンディングボックス座標 (x1, y1, x2, y2)。

        Returns:
            bool: 位置関係が妥当であれば`True`、そうでなければ`False`。
        """
        px, py = person_center
        sx, sy = suitcase_center
        person_bottom = person_bbox[3]

        if sy < py and sx > px and (sx - px) > 50:
            print(f"    ❌ 位置関係異常: スーツケースが人の右斜め上にある")
            return False  # スーツケースが人物の不自然な位置（右上など）にある場合は無効

        if sy < person_bottom - 200:
            print(f"    ❌ 位置関係異常: スーツケースの位置が高すぎる")
            return False  # スーツケースの位置が高すぎる場合は無効（持ち上げていない限り）

        dx = sx - px
        dy = sy - py

        if abs(dx) > 0:
            angle = math.atan2(dy, dx) * 180 / math.pi
            if not (-45 <= angle <= 225):
                print(f"    ❌ 位置関係異常: スーツケースの角度が不適切 {angle:.1f}度")
                return False  # 水平方向の角度が不適切な場合は無効（真後ろなどを想定）

        return True

    def _record_matching_history(self, suitcase_id, person_id):
        """マッチング履歴の記録
        人物IDとスーツケースIDのペアを、マッチング履歴辞書に記録します。
        この関数は、過去に成功した組み合わせを追跡するために利用されます。

        - 履歴は`{suitcase_id: {person_id1, person_id2, ...}}`の形式で管理されます。
        - この関数は、記録のみを行い、重複のチェックやその他のロジックは含みません。

        Args:
            suitcase_id (str): スーツケースのID。
            person_id (str): 人物のID。
    """
        if suitcase_id not in self.suitcase_match_history:
            self.suitcase_match_history[suitcase_id] = set()
        self.suitcase_match_history[suitcase_id].add(person_id)  # スーツケースに人物IDを記録

    def check_repeated_matching(self, suitcase_id, person_id):
        """繰り返しマッチングのチェックと履歴の記録
        与えられた人物とスーツケースのペアが、過去のマッチング履歴に
        存在するかをチェックします。この関数は、重複したアラートの送信や
        不必要な処理を防ぐために設計されています。

        .. warning::
            現在の実装では、**重複チェックは行われず**、常に`False`を返します。
            この関数は、マッチング履歴を記録する目的でのみ使用されます。
            将来的に、重複処理を適切に防ぐために、実際のチェックロジックを
            実装する必要があります。

        Args:
            suitcase_id (str): スーツケースのID。
            person_id (str): 人物のID。

        Returns:
            bool: **常に`False`を返します。**
        """
        self._record_matching_history(suitcase_id, person_id)  # 履歴を記録
        return False  # 常にFalseを返す（警告の通り、現在チェックは行わない）


    def _update_velocity_and_static(self, obj_map_prev, static_map, obj_id, curr_center):
        """物体の速度と静止状態を更新します。

        この関数は、毎フレーム呼び出され、物体（人物またはスーツケース）の
        速度を計算し、その静止フレーム数を追跡します。これにより、後続の
        静止判定ロジックのための基礎データを提供します。

        Args:
            obj_map_prev (dict): 前のフレームでの物体の中心座標を格納する辞書。
            static_map (dict): 物体の静止フレーム数をカウントする辞書。
            obj_id (str): 物体を一意に識別するID。
            curr_center (tuple[float, float]): 現在のフレームでの物体の中心座標 (x, y)。

        Returns:
            float: 計算された物体の速度。
        """
        px, py = obj_map_prev.get(obj_id, curr_center)
        vx, vy = curr_center[0] - px, curr_center[1] - py
        obj_map_prev[obj_id] = curr_center
        speed = (vx*vx + vy*vy) ** 0.5
        if speed <= self.VEL_STATIC_TH:
            static_map[obj_id] = static_map.get(obj_id, 0) + 1  # 静止している場合、連続フレーム数を増加
        else:
            static_map[obj_id] = 0  # 移動している場合、連続フレーム数をリセット
        return speed

    def _is_static_long_enough(self, person_id, suitcase_id):
        """人物とスーツケースが十分に長く静止しているかを判定します。

        この関数は、両方の物体が、予め設定された静止フレーム数（`STATIC_STREAK_REQ`）を
        満たしているかどうかをチェックします。この静止状態の判断は、マッチングの信頼性を
        高めるための重要なフィルタリング条件として利用されます。

        Args:
            person_id (str): 人物のID。
            suitcase_id (str): スーツケースのID。

        Returns:
            bool: 両方の物体が十分な期間静止していれば`True`、そうでなければ`False`。
        """
        p_streak = self.person_static_streak.get(person_id, 0)
        s_streak = self.suitcase_static_streak.get(suitcase_id, 0)
        # 両方の静止フレーム数が要求フレーム数以上であるかをチェック
        return (p_streak >= self.STATIC_STREAK_REQ) and (s_streak >= self.STATIC_STREAK_REQ)

    def _strong_posture_evidence(self, person, suitcase):
        """人物の姿勢から強い物理的接触の証拠を検出します。

        この関数は、姿勢推定によって得られた人物のキーポイント情報（特に手首や腰）を
        利用し、人物とスーツケースが物理的に接触している、あるいは強く関連している
        可能性を判断します。これは、単純な距離だけでは判断できない「引く」「持つ」
        といった行為を検出するために使用されます。

        - **機能:**
            1.  人物のキーポイントデータを取得し、信頼度が低いポイント（0.3未満）は除外します。
            2.  各有効なキーポイント（左右の手首、股関節）から、スーツケースの
                バウンディングボックスの上辺までの最短距離を計算します。
            3.  この最短距離が、設定されたしきい値（`STRONG_POSTURE_DIST`=100）以下である
                キーポイントの数をカウントします。
            4.  有効なキーポイントのうち、この条件を満たすものが半分以上あれば、
                人物とスーツケース間に強い関連性があると判断し、`True`を返します。

        - **入力:**
            - `person` (dict): 検出された人物のデータ。`keypoints`キー（[x, y, 信頼度]のリスト）を
            含んでいる必要があります。
            - `suitcase` (dict): 検出されたスーツケースのデータ。`bbox`キー（[x1, y1, x2, y2]）を
            含んでいる必要があります。

        - **出力:**
            - `bool`: 強い姿勢のエビデンスがあれば`True`、そうでなければ`False`。
        """
        kp = person.get('keypoints', None)
        if kp is None:
            return False

        key_points = [9, 10, 11, 12]  # 左右手首，左右股関節
        x1, y1, x2, y2 = suitcase['bbox']
        
        close_points = 0
        total_valid_points = 0
        
        for idx in key_points:
            if idx < len(kp) and len(kp[idx]) >= 3:
                xh, yh, conf = kp[idx][0], kp[idx][1], kp[idx][2]
                if conf > 0.3:
                    total_valid_points += 1
                    # 箱の上辺に対して最短距離の参照点を構成
                    if xh < x1:
                        cx, cy = x1, y1
                    elif xh > x2:
                        cx, cy = x2, y1
                    else:
                        cx, cy = xh, y1
                    
                    dist = ((xh - cx)**2 + (yh - cy)**2) ** 0.5
                    if dist <= self.STRONG_POSTURE_DIST:
                        close_points += 1
        
        return total_valid_points > 0 and (close_points / total_valid_points) >= 0.5

    def _is_adult_like(self, person):
        """人物が大人らしい体格であるかを判断します。

        この関数は、人物のバウンディングボックスの高さが、フレーム全体の高さに
        対して占める割合を計算し、その比率が事前に設定された大人の基準値
        （`ADULT_HEIGHT_RATIO`）以上であるかを判断します。
        この判断は、人物のカメラからの相対的な距離や体格を間接的に推定するために
        利用され、動的なマッチング距離しきい値の計算に組み込まれます。

        Args:
            person (dict): 検出された人物のデータ。`bbox`キー（[x1, y1, x2, y2]）を
                        含んでいる必要があります。

        Returns:
            bool: 人物が大人らしい体格と判断されれば`True`、そうでなければ`False`。
        """
        ph = person['bbox'][3] - person['bbox'][1]  # 人物BBoxの高さ
        # 高さの比率が大人判定の閾値以上であるかをチェック
        return (ph / max(self.frame_height, 1)) >= self.ADULT_HEIGHT_RATIO

    def is_suitcase_near_person(self, person, suitcase):
        """人物とスーツケースが関連しているかを多角的に判断します。

        この関数は、人物とスーツケースのペアに対して、距離、位置関係、動態、
        過去の履歴、姿勢、そして特定のフィルタリングルールを組み合わせて、
        マッチングの妥当性を総合的に評価します。

        - **全体のフロー:**
        1.  **初期フィルタリング:** スーツケースがゲートエリアにいる場合、
            即座にマッチングを無効と判断し、詳細な分析をスキップします。
        2.  **基本情報の更新:** 人物とスーツケースの速度と静止状態を更新します。
        3.  **主要な特徴量計算:** 距離、位置、姿勢、履歴、初回接近のスコアを計算します。
        4.  **行動に基づくフィルタリング:**
            -   両者が長時間静止しており、かつ所有者でなく、強い姿勢の証拠もない場合、
                新規のマッチング構築を`allow_new_build=False`でブロックします。
            -   同様に、静止したスーツケースに対して子供が近づいた場合も、
                履歴ボーナスが低い場合にマッチングをブロックします。
        5.  **最終的な判断:** 距離、垂直方向の位置関係、および上記で設定された
            `allow_new_build`フラグに基づいて、最終的なマッチングの有効性を決定します。

        Args:
            person (dict): 検出された人物のデータ。`id`, `center`, `bbox`などのキーを
                        含んでいる必要があります。
            suitcase (dict): 検出されたスーツケースのデータ。`id`, `center`, `bbox`, `type`などの
                            キーを含んでいる必要があります。

        Returns:
            dict: マッチングの評価結果を格納した辞書。
                `overall_match` (bool) で最終的な判断を示し、その他のキーで各評価要素の
                詳細な値を提供します。
        """

        self.gate_filter_stats['total_attempts'] += 1  # 総マッチング試行回数をカウント

        if self.is_in_gate_area(suitcase['bbox']):
            self.gate_filter_stats['area_filtered'] += 1  # ゲートエリアでフィルタリングされた回数をカウント
            return {
                'overall_match': False,
                'distance': 9999,
                'pose_score': -1.0,
                'first_approach_score': 0.0,
                'filter_reason': 'confirmed_gate'  # ゲートエリアフィルタリングによる除外
            }

        # 速度・静止状態を更新
        person_speed = self._update_velocity_and_static(
            self.prev_person_center, self.person_static_streak,
            person['id'], tuple(person['center'])
        )
        suitcase_speed = self._update_velocity_and_static(
            self.prev_suitcase_center, self.suitcase_static_streak,
            suitcase['id'], tuple(suitcase['center'])
        )

        dynamic_threshold = self._calculate_dynamic_distance_threshold(person)  # 動的距離閾値を計算
        distance = self.calculate_distance(person['center'], suitcase['center'])  # 距離を計算
        distance_match = distance <= dynamic_threshold  # 距離が閾値内であるか

        person_bottom = person['bbox'][3]
        suitcase_top = suitcase['bbox'][1]
        vertical_ok = suitcase_top <= person_bottom + 80  # スーツケースの上端が人物の足元付近より上にあるか

        item_type = suitcase.get('type', 'suitcase')  # 物品タイプを取得
        pose_score = self.calculate_pose_score(person, suitcase, item_type)
        first_approach_score = self._calculate_first_approach_score(person['id'], suitcase['id'], distance)
        position_memory_bonus = self._get_position_memory_bonus(person, suitcase)

        self._record_matching_history(suitcase['id'], person['id'])   # 履歴を記録

        is_both_static_long = self._is_static_long_enough(person['id'], suitcase['id'])  # 両者が長く静止しているか
        sid = suitcase['id']
        pid = person['id']
        is_owner_pair = (self.suitcase_owner.get(sid) == pid)  # 所有者と一致するか
        strong_pose = self._strong_posture_evidence(person, suitcase)  # 強い姿勢の証拠があるか

        suitcase_is_static = (self.suitcase_static_streak.get(sid, 0) >= self.STATIC_STREAK_REQ)
        height_category = self._estimate_person_height_category(person)

        allow_new_build = True  # マッチングの新規構築を許可するか

        # フィルタリングA: 両者静止、かつ所有者でも強い姿勢でもない場合、新規マッチングをブロック
        if is_both_static_long and not is_owner_pair and not strong_pose:
            allow_new_build = False

        # フィルタリングB: スーツケース静止、人物が子供、所有者でも強い姿勢でもない、かつ履歴ボーナスが低い場合、ブロック
        if (self.ADULT_REQUIRED_WHEN_SUITCASE_STATIC and suitcase_is_static 
            and not is_owner_pair and height_category == 'child' and not strong_pose):
            if position_memory_bonus < 0.3:
                allow_new_build = False

        # 最終判断: 距離、垂直関係、新規構築許可がすべてTrueの場合にマッチ
        overall = (distance_match and vertical_ok and allow_new_build)

        return {
            'overall_match': overall,
            'distance': distance,
            'pose_score': pose_score,
            'first_approach_score': first_approach_score,
            'position_memory_bonus': position_memory_bonus,
            'height_category': height_category,
            'dynamic_threshold_used': dynamic_threshold,
            'filter_reason': 'normal_matching',
            'person_speed': person_speed,
            'suitcase_speed': suitcase_speed,
            'both_static_gate': (is_both_static_long and not is_owner_pair and not strong_pose),
            'adult_gate_block': (self.ADULT_REQUIRED_WHEN_SUITCASE_STATIC and suitcase_is_static
                                 and not is_owner_pair and height_category == 'child' and not strong_pose)
        }

    def _calculate_first_approach_score(self, person_id, suitcase_id, current_distance):
        """
        人物がスーツケースに最初に接近したことに対するスコアを計算します。

        この関数は、人物とスーツケースのペアが、過去の接触履歴においてどのような関係にあるかを
        評価し、その信頼性をスコアとして返します。これは、新しい所有者が現れたり、
        元の所有者が去ったりする状況を判断するための重要なヒューリスティックです。

        Args:
            person_id (str): 評価対象の人物ID。
            suitcase_id (str): 評価対象のスーツケースID。
            current_distance (float): 現在の人物とスーツケース間の距離。

        Returns:
            float: 接近の信頼性を示す0.0から0.5までのスコア。
        """
        if current_distance > self.distance_threshold * 1.5:
            return 0.0

        if suitcase_id not in self.first_approach_history:
            self.first_approach_history[suitcase_id] = person_id  # 初回接近者を記録
            self.approach_timestamps[(person_id, suitcase_id)] = self._get_current_frame()
            return 0.5  # 初回接近ボーナス（高スコア）

        first_person = self.first_approach_history[suitcase_id]
        if person_id == first_person:
            return 0.4   # 初回接近者と同一人物であればボーナス（中スコア）
        else:
            if self._is_first_person_gone(suitcase_id, first_person):
                self.first_approach_history[suitcase_id] = person_id
                self.approach_timestamps[(person_id, suitcase_id)] = self._get_current_frame()
                return 0.3  # 新しい接近者ボーナス（やや低スコア）
            else:
                return 0.0 # 元の人物がまだ関連している場合はボーナスなし

    def _is_first_person_gone(self, suitcase_id, first_person_id):
        """
        最初に接近した人物が、もはやそのスーツケースの所有者ではないかを判断します。

        この関数は、過去のマッチング履歴を参照し、元の所有者とスーツケースの間の
        関連性が薄れたかどうかをチェックします。

        Args:
            suitcase_id (str): スーツケースID。
            first_person_id (str): 最初に接近した人物のID。

        Returns:
            bool: 最初の人物が去ったと見なされれば`True`、そうでなければ`False`。
        """
        if first_person_id in self.match_history:
            if suitcase_id in self.match_history[first_person_id]:
                recent_score = self.match_history[first_person_id][suitcase_id]
                return recent_score < 1.0
        return True

    def _get_current_frame(self):
        """
        現在のフレーム数を取得するユーティリティ関数です。

        この関数は、時間の経過を追跡するために使用され、通常はメインループの
        フレームカウンターと同期して呼び出されます。ここでは、タイムスタンプ
        辞書のキーの数でフレーム数を代用しています。

        Args:
            なし

        Returns:
            int: 現在のフレーム数を表す整数。
        """
        return len(self.approach_timestamps)

    # 従来の calculate_pose_score を置き換え
    def calculate_pose_score(self, person, suitcase, item_type="suitcase"):
        """
        改良した姿勢解析：上肢と下肢の統合評価
        """
        if 'keypoints' not in person or person['keypoints'] is None:
            return 0.0

        keypoints = person['keypoints']
        suitcase_bbox = suitcase['bbox']

        if item_type == "backpack":
            return self._analyze_backpack_pose(person, suitcase)
        elif item_type == "handbag":
            return self._analyze_handbag_pose(person, suitcase)
        else:  # suitcase またはデフォルト
            return self._analyze_suitcase_pose(person, suitcase)
        
    
    def _analyze_backpack_pose(self, person, backpack):
        """
        バックパック姿勢解析：主に胴体中心からバックパックまでの距離に基づく
        """
        if 'keypoints' not in person or person['keypoints'] is None:
            return 0.0
        
        keypoints = person['keypoints']
        
        # 胴体中心点を計算
        torso_center = self._get_torso_center(keypoints)
        if torso_center is None:
            return 0.0
        
        backpack_center = backpack['center']
        
        # 距離を計算
        distance = self._calculate_distance_2d(torso_center, backpack_center)
        
        # 距離スコア（バックパックはスーツケースより身体に密着）
        if distance < 60:    # 非常に近い：背負っている可能性が高い
            return 0.4
        elif distance < 120:  # 中程度の距離：関連の可能性
            return 0.25
        elif distance < 200:  # やや遠いが可能性あり
            return 0.1
        else:                # 遠すぎ：可能性が低い
            return 0.0

    def _get_torso_center(self, keypoints):
        """
        胴体中心点を計算：肩の中点を優先，代替として腰（股関節）の中点
        """
        try:
            left_shoulder = keypoints[5]   # 左肩
            right_shoulder = keypoints[6]  # 右肩
            left_hip = keypoints[11]       # 左股関節
            right_hip = keypoints[12]      # 右股関節
            
            # まず肩の中点を使用
            if (self._is_keypoint_valid(left_shoulder) and 
                self._is_keypoint_valid(right_shoulder)):
                return [
                    (left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2
                ]
            
            # 代替：股関節の中点
            elif (self._is_keypoint_valid(left_hip) and 
                self._is_keypoint_valid(right_hip)):
                return [
                    (left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2
                ]
            
            return None
            
        except (IndexError, TypeError):
            return None

    def _analyze_handbag_pose(self, person, handbag):
        """
        ハンドバッグ姿勢解析：暫定の簡易ロジック
        """
        # ひとまず中立的なスコアを返す。後で精緻化可能。
        return 0.1

    def _analyze_suitcase_pose(self, person, suitcase):
        """
        スーツケース姿勢解析：既存の上肢/下肢の複合分析を使用
        """
        if 'keypoints' not in person or person['keypoints'] is None:
            return 0.0

        keypoints = person['keypoints']
        suitcase_bbox = suitcase['bbox']
        
        # 自適応重み付けのためにスーツケースの特徴量を計算
        suitcase_width = suitcase_bbox[2] - suitcase_bbox[0]
        suitcase_height = suitcase_bbox[3] - suitcase_bbox[1]
        aspect_ratio = suitcase_height / max(suitcase_width, 1)
        
        # 箱の形状から重みを推定
        if aspect_ratio > 1.3:  # 縦長（ドラッグ型の可能性）
            upper_weight = 0.7
            lower_weight = 0.3
        else:  # 比較的スクエア（プッシュ型の可能性）
            upper_weight = 0.5
            lower_weight = 0.5
        
        # 上肢の分析
        upper_score = self._analyze_upper_limbs(keypoints, suitcase_bbox)
        
        # 下肢の分析
        lower_score = self._analyze_lower_limbs(keypoints, suitcase_bbox)
        
        # 融合スコア
        final_pose_score = upper_weight * upper_score + lower_weight * lower_score

        return final_pose_score
        

    # 以下の新メソッドをすべて ImprovedRealtimePersonSuitcaseMatcher クラスに追加

    def _analyze_upper_limbs(self, keypoints, suitcase_bbox):
        """上肢姿勢解析：手部と箱上辺の関係"""
        try:
            left_wrist = keypoints[9]   # 左手首
            right_wrist = keypoints[10] # 右手首
            left_elbow = keypoints[7]   
            right_elbow = keypoints[8]  
            left_shoulder = keypoints[5]  
            right_shoulder = keypoints[6] 
            
            scores = []
            
            # 左手の分析
            if self._is_keypoint_valid(left_wrist) and self._is_keypoint_valid(left_elbow):
                left_score = self._evaluate_arm_pose(
                    left_shoulder, left_elbow, left_wrist, suitcase_bbox
                )
                scores.append(left_score)
            
            # 右手の分析
            if self._is_keypoint_valid(right_wrist) and self._is_keypoint_valid(right_elbow):
                right_score = self._evaluate_arm_pose(
                    right_shoulder, right_elbow, right_wrist, suitcase_bbox
                )
                scores.append(right_score)
            
            if not scores:
                return 0.0
                
            # より高い方の手のスコアを採用（通常は片手でドラッグ）
            return max(scores)
            
        except (IndexError, TypeError):
            return 0.0

    def _analyze_lower_limbs(self, keypoints, suitcase_bbox):
        """下肢姿勢解析：脚と箱の距離関係"""
        try:
            left_knee = keypoints[13]   
            right_knee = keypoints[14]  
            left_ankle = keypoints[15]  
            right_ankle = keypoints[16] 
            
            suitcase_center = [
                (suitcase_bbox[0] + suitcase_bbox[2]) / 2,
                (suitcase_bbox[1] + suitcase_bbox[3]) / 2
            ]
            
            leg_scores = []
            
            # 左脚の分析
            if self._is_keypoint_valid(left_knee):
                left_knee_dist = self._calculate_distance_2d(left_knee[:2], suitcase_center)
                left_leg_score = self._evaluate_leg_distance(left_knee_dist)
                leg_scores.append(left_leg_score)
            
            if self._is_keypoint_valid(left_ankle):
                left_ankle_dist = self._calculate_distance_2d(left_ankle[:2], suitcase_center)
                left_ankle_score = self._evaluate_leg_distance(left_ankle_dist)
                leg_scores.append(left_ankle_score)
            
            # 右脚の分析
            if self._is_keypoint_valid(right_knee):
                right_knee_dist = self._calculate_distance_2d(right_knee[:2], suitcase_center)
                right_leg_score = self._evaluate_leg_distance(right_knee_dist)
                leg_scores.append(right_leg_score)
                
            if self._is_keypoint_valid(right_ankle):
                right_ankle_dist = self._calculate_distance_2d(right_ankle[:2], suitcase_center)
                right_ankle_score = self._evaluate_leg_distance(right_ankle_dist)
                leg_scores.append(right_ankle_score)
            
            if not leg_scores:
                return 0.0
                
            # 最良の脚スコアを返す
            return max(leg_scores)
            
        except (IndexError, TypeError):
            return 0.0

    def _evaluate_arm_pose(self, shoulder, elbow, wrist, suitcase_bbox):
        """手腕姿勢の妥当性を評価"""
        if not all([self._is_keypoint_valid(kp) for kp in [shoulder, elbow, wrist]]):
            return 0.0
        
        # 手首から箱上辺までの距離（既存ロジックを維持）
        wrist_to_top_dist = self._distance_to_top_edge(wrist, suitcase_bbox)
        distance_score = self._evaluate_hand_distance(wrist_to_top_dist)
        
        # 追加：手腕姿勢の妥当性分析
        posture_score = self._evaluate_arm_posture(shoulder, elbow, wrist, suitcase_bbox)
        
        # 距離と姿勢スコアを融合
        return 0.6 * distance_score + 0.4 * posture_score

    def _evaluate_arm_posture(self, shoulder, elbow, wrist, suitcase_bbox):
        """手腕姿勢の妥当性を分析"""
        try:
            # 上腕と前腕のベクトルを計算
            upper_arm = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]
            lower_arm = [wrist[0] - elbow[0], wrist[1] - elbow[1]]
            
            # 手首から箱中心へのベクトル
            suitcase_center = [(suitcase_bbox[0] + suitcase_bbox[2]) / 2,
                            (suitcase_bbox[1] + suitcase_bbox[3]) / 2]
            wrist_to_suitcase = [suitcase_center[0] - wrist[0], suitcase_center[1] - wrist[1]]
            
            # 前腕と箱方向ベクトルの整合性
            alignment_score = self._calculate_vector_alignment(lower_arm, wrist_to_suitcase)
            
            # 手腕の伸展度（伸ばしているほどドラッグの可能性）
            arm_extension = self._calculate_arm_extension(shoulder, elbow, wrist)
            
            return 0.6 * alignment_score + 0.4 * arm_extension
            
        except:
            return 0.0
        

    def _distance_to_top_edge(self, hand_point, suitcase_bbox):
        """手部キーポイントからスーツケース上辺までの最短距離を計算"""
        hand_x, hand_y = hand_point[0], hand_point[1]
        x1, y1, x2, y2 = suitcase_bbox

        if hand_x < x1:
            closest_x, closest_y = x1, y1
        elif hand_x > x2:
            closest_x, closest_y = x2, y1
        else:
            closest_x, closest_y = hand_x, y1

        dx = hand_x - closest_x
        dy = hand_y - closest_y
        distance = (dx**2 + dy**2)**0.5

        return distance

    def _evaluate_leg_distance(self, distance):
        """脚と箱の距離の妥当性を評価"""
        if distance < 80:   # 非常に近い：並走プッシュの可能性
            return 0.3
        elif distance < 150: # 中距離：妥当な範囲
            return 0.2
        elif distance < 250: # やや遠いが関連の可能性
            return 0.1
        else:               # 遠すぎ：関連が低い
            return 0.0

    def _calculate_vector_alignment(self, vec1, vec2):
        """2つのベクトルの整合度を計算"""
        try:
            # ベクトル長
            len1 = (vec1[0]**2 + vec1[1]**2)**0.5
            len2 = (vec2[0]**2 + vec2[1]**2)**0.5
            
            if len1 == 0 or len2 == 0:
                return 0.0
            
            # コサイン類似度
            dot_product = vec1[0]*vec2[0] + vec1[1]*vec2[1]
            cos_sim = dot_product / (len1 * len2)
            
            # 0-1スコアに変換（より整合で高スコア）
            alignment = (cos_sim + 1) / 2
            return min(max(alignment, 0), 1)
            
        except:
            return 0.0

    def _calculate_arm_extension(self, shoulder, elbow, wrist):
        """手腕の伸展度を計算"""
        try:
            # 肩～手首の直線距離
            direct_dist = self._calculate_distance_2d(shoulder[:2], wrist[:2])
            
            # 肩→肘→手首の経路距離
            path_dist = (self._calculate_distance_2d(shoulder[:2], elbow[:2]) + 
                        self._calculate_distance_2d(elbow[:2], wrist[:2]))
            
            if path_dist == 0:
                return 0.0
                
            # 伸展比（1に近いほど腕が伸びている）
            extension_ratio = direct_dist / path_dist
            
            # 適度な伸展が最良（完全伸展/完全屈曲はいずれも最適ではない）
            if 0.7 <= extension_ratio <= 0.9:
                return 0.3
            elif 0.6 <= extension_ratio <= 0.95:
                return 0.2
            else:
                return 0.1
                
        except:
            return 0.0

    def _is_keypoint_valid(self, keypoint, confidence_threshold=0.3):
        """キーポイントが有効かをチェック"""
        try:
            return len(keypoint) >= 3 and keypoint[2] >= confidence_threshold
        except:
            return False

    def _calculate_distance_2d(self, point1, point2):
        """2Dのユークリッド距離を計算"""
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        return (dx**2 + dy**2)**0.5


    def _evaluate_hand_distance(self, distance):
        """手と箱の距離の妥当性を評価"""
        if distance < 60:
            return 0.3
        elif distance < 120:
            return 0.15
        elif distance < 200:
            return 0.0
        elif distance < 300:
            return -0.1
        else:
            return -0.2

    def update_matches(self, frame_data):
        """人物とスーツケースのマッチングを毎フレーム更新します。

        この関数は、アルゴリズム全体の中心であり、以下の主要なステップを実行します。
        1.  **候補生成**: フレーム内のすべての人とスーツケースのペアを評価し、マッチングの可能性が高いペアを抽出してスコアを計算します。
        2.  **割り当て解決**: 候補リストから最適な一対一のマッチングセットを決定します。
        3.  **状態更新**: 決定されたマッチング情報を用いて、システムの長期的な状態（履歴、所有権、メモリ）を更新・維持します。

        Args:
            frame_data (dict): 現在のフレームの検出データ。`'persons'`と`'suitcases'`のキーを
                            含み、それぞれのリストを格納します。

        Returns:
            dict: 現在のフレームで確定された人物IDとスーツケースIDのペア。
        """
        self.current_frame_count += 1
        all_possible_matches = []

        # ステップ1: 候補生成とスコアリング（総当たりで候補ペアを評価）
        for person in frame_data['persons']:
            for suitcase in frame_data['suitcases']:
                # 各ペアの適格性と個別スコアを評価
                match_result = self.is_suitcase_near_person(person, suitcase)

                if match_result['overall_match']:
                    # === 最終スコアの計算と統合 ===

                    # 距離スコアを正規化（距離が近いほど高スコア）
                    distance_score = 1.0 / (1.0 + match_result['distance'] / 100)
                    pose_score = match_result['pose_score']
                    first_approach_score = match_result['first_approach_score']
                    position_memory_bonus = match_result.get('position_memory_bonus', 0.0)

                    # 既存の所有権ボーナスを計算 (所有権の維持に最も重要)
                    owner_bonus = 0.0
                    if self.suitcase_owner.get(suitcase['id']) == person['id']:
                        owner_bonus = self.OWNER_BONUS

                    # 人物属性（大人の体格）ボーナスを計算
                    adult_bonus = 0.0
                    if match_result.get('height_category') == 'adult':
                        adult_bonus = 0.08

                    # すべてのスコアを重み付けして最終的な信頼度を算出
                    final_score = (
                        0.9 * distance_score +
                        1.0 * pose_score +
                        0.8 * first_approach_score +
                        1.2 * owner_bonus +
                        0.8 * position_memory_bonus +
                        0.3 * adult_bonus
                    )

                    # 有効な候補ペアをリストに追加
                    all_possible_matches.append({
                        'person_id': person['id'],
                        'suitcase_id': suitcase['id'],
                        'distance': match_result['distance'],
                        'pose_score': pose_score,
                        'first_approach_score': first_approach_score,
                        'position_memory_bonus': position_memory_bonus,
                        'score': final_score,
                        'person_obj': person,
                        'suitcase_obj': suitcase
                    })
        # ステップ2: 割り当て解決 (一対一のマッチングの確定)
        # 貪欲法により、最もスコアの高い、互いに重複しないペアリングを決定
        current_frame_matches = self._solve_assignment_problem(all_possible_matches)

        # ステップ3: 履歴と長期状態の更新
        # 3.1: 短期履歴 (match_history) の更新
        # マッチングが連続しているフレーム数を加算
        for person_id, suitcase_id in current_frame_matches.items():
            if person_id not in self.match_history:
                self.match_history[person_id] = {}
            if suitcase_id not in self.match_history[person_id]:
                self.match_history[person_id][suitcase_id] = 0
            self.match_history[person_id][suitcase_id] += 1

        # 3.2: 確定マッチ (confirmed_matches) の識別
        # 最小連続フレーム数 (min_match_frames) を超えたペアを確定
        self.confirmed_matches = {}
        for person_id, suitcase_matches in self.match_history.items():
            if suitcase_matches:
                best_suitcase = max(suitcase_matches.items(), key=lambda x: x[1])
                suitcase_id, match_count = best_suitcase
                if match_count >= self.min_match_frames:
                    self.confirmed_matches[person_id] = suitcase_id

        # 3.3: 矛盾の解消 (一つの荷物に複数人がマッチした場合の最終調整)
        self.confirmed_matches = self._ensure_one_to_one_confirmed_matches(self.confirmed_matches)

        # 3.4: 記憶と所有権の確立
        for person_id, suitcase_id in self.confirmed_matches.items():
            person_obj = next((p for p in frame_data['persons'] if p['id'] == person_id), None)
            suitcase_obj = next((s for s in frame_data['suitcases'] if s['id'] == suitcase_id), None)
            
            # 成功したマッチングの位置関係を長期メモリに保存
            if person_obj and suitcase_obj:
                self._update_position_memory(person_obj, suitcase_obj, match_confirmed=True)

        for p_id, s_id in self.confirmed_matches.items():
            self.suitcase_owner[s_id] = p_id
            self.suitcase_owner_ttl[s_id] = self.OWNER_TTL

         # 3.5: TTLの管理と期限切れ所有権の削除
        for s_id in list(self.suitcase_owner_ttl.keys()):
            self.suitcase_owner_ttl[s_id] -= 1
            if self.suitcase_owner_ttl[s_id] <= 0:
                self.suitcase_owner.pop(s_id, None)
                self.suitcase_owner_ttl.pop(s_id, None)

        # 3.6: 履歴データのクリーンアップ
        self._cleanup_position_memory()
        self._cleanup_history()
        
        return current_frame_matches

    def _solve_assignment_problem(self, all_possible_matches):
        """現在のフレームにおける人物とスーツケースの最適な一対一マッチングを解決します。

        この関数は、与えられたすべての一致候補リストから、各人物が1つのスーツケースに、
        そして各スーツケースが1人の人物に割り当てられるように、最適なペアリングを決定します。
        ここでは、スコアの高いマッチを優先的に採用する**貪欲法（Greedy Algorithm）**が使用されます。

        - **動作原理**:
            1.  入力リスト`all_possible_matches`を、計算された`'score'`に基づいて降順にソートします。
            2.  ソートされたリストを先頭から順に走査します。
            3.  各ペアについて、人物とスーツケースのどちらもまだ割り当てられていない場合、
                そのペアを最終的なマッチングとして確定します。
            4.  一度割り当てが確定した人物およびスーツケースは、以降のチェックでは
                スキップされます。

        Args:
            all_possible_matches (list[dict]): `is_suitcase_near_person`関数によって
                                            評価された、すべての人物-スーツケースの
                                            候補ペアとスコアのリスト。

        Returns:
            dict: 最も信頼性の高い一対一のマッチングセット。
                キーは人物ID、値はスーツケースIDとなります。
        """
        if not all_possible_matches:
            return {}

        all_possible_matches.sort(key=lambda x: x['score'], reverse=True)

        assigned_persons = set()
        assigned_suitcases = set()
        final_matches = {}

        for match in all_possible_matches:
            person_id = match['person_id']
            suitcase_id = match['suitcase_id']

            if person_id not in assigned_persons and suitcase_id not in assigned_suitcases:
                final_matches[person_id] = suitcase_id
                assigned_persons.add(person_id)
                assigned_suitcases.add(suitcase_id)

        return final_matches

    def _ensure_one_to_one_confirmed_matches(self, confirmed_matches):
        """長期確定マッチングセットが厳格な一対一関係を保つことを保証します。

        この関数は、マッチング履歴の累積により発生しうる「一つのスーツケースが複数の人物に
        マッチングされる」という矛盾を解決します。特にIDの切り替わりなどによって引き起こされる
        不整合を修正し、長期的な所有権関係の信頼性を維持します。

        - **動作原理**:
            1.  入力された確定マッチングセットを、スーツケースIDをキー、人物IDのリストを値とする
                一時的な辞書に変換します。これにより、多対一の関係（複数の人物が一つのスーツケースに
                紐づけられている状況）を特定します。
            2.  各スーツケースについて、紐づいている人物が一人だけの場合、その関係をそのまま
                クリーンなマッチングリストに移動します。
            3.  紐づいている人物が複数いる場合、`match_history`を参照して、そのスーツケースとの
                マッチングが**最も長く続いている**人物を特定します。
            4.  最も長いマッチング履歴を持つ人物を、そのスーツケースの唯一の所有者として確定します。

        Args:
            confirmed_matches (dict): `match_history`から抽出された、現在の確定マッチングセット。

        Returns:
            dict: 矛盾が解消され、すべてのペアが厳格な一対一関係である、クリーンなマッチングセット。
        """
        # ステップ1: 確定マッチングを「スーツケース中心」のビューに変換し、多対一の矛盾を特定
        suitcase_to_persons = {}
        for person_id, suitcase_id in confirmed_matches.items():
            if suitcase_id not in suitcase_to_persons:
                suitcase_to_persons[suitcase_id] = []
            suitcase_to_persons[suitcase_id].append(person_id)

        cleaned_matches = {}

        # ステップ2: 各スーツケースの矛盾を解決
        for suitcase_id, person_list in suitcase_to_persons.items():
            if len(person_list) == 1:
                # 矛盾なし: そのままクリーンなリストに追加 (一対一の関係)
                cleaned_matches[person_list[0]] = suitcase_id
            else:
                # 矛盾あり: 複数の人物が同じスーツケースに紐づいている (IDスイッチなどの結果)
                best_person = None
                max_match_count = 0

                 # ステップ3: マッチング履歴を比較し、最も持続的な人物を特定
                for person_id in person_list:
                    # match_historyから、人物とスーツケースの連続マッチング回数を取得
                    match_count = self.match_history[person_id].get(suitcase_id, 0)

                    # 最長のマッチング回数を記録した人物を「真の所有者」と見なす
                    if match_count > max_match_count:
                        max_match_count = match_count
                        best_person = person_id

                # ステップ4: 最も持続的な人物を唯一の所有者として確定
                if best_person:
                    cleaned_matches[best_person] = suitcase_id

        return cleaned_matches

    def _cleanup_history(self):
        """
        マッチング履歴を整理し、古くなった情報を削除します。

        この関数は、以下の2つの主要なクリーンアップタスクを実行します。
        1.  **短期履歴の減衰と削除**: `match_history`内の各マッチングペアの連続フレーム数を
            徐々に減らします。カウントが0.5未満になったペアは削除され、これにより、
            短期間しか続かなかった不確実なマッチングが履歴に残り続けるのを防ぎます。
        2.  **長期履歴の同期と削除**: 100フレームごとに、`match_history`に存在しない
            スーツケース（つまり、もはやどの人物ともマッチングしていないスーツケース）
            を`suitcase_match_history`から削除します。これは、過去の所有者が去った後に
            取り残された履歴をクリーンアップするのに役立ちます。

        この処理は、システムのメモリを最適化し、古くなった情報が新たなマッチングの
        判断に誤った影響を与えることを防ぎます。

        Args:
            self (object): インスタンス自身。`match_history`、`suitcase_match_history`、
                        `_get_current_frame`、`_cleanup_first_approach_history`に
                        アクセスするために必要です。
        """
        # 短期履歴の減衰処理を開始
        for person_id in list(self.match_history.keys()):
            for suitcase_id in list(self.match_history[person_id].keys()):
                self.match_history[person_id][suitcase_id] = max(0,
                    self.match_history[person_id][suitcase_id] - 0.1)

                # カウントが閾値(0.5)未満になったペアは履歴から削除
                if self.match_history[person_id][suitcase_id] < 0.5:
                    del self.match_history[person_id][suitcase_id]

            # その人物に関連付けられたスーツケースがなくなった場合、人物のエントリも削除
            if not self.match_history[person_id]:
                del self.match_history[person_id]

        # 初回接近履歴のクリーンアップを実行
        self._cleanup_first_approach_history()

        # 長期履歴のクリーンアップ（100フレームごと）
        current_frame = self._get_current_frame()
        if current_frame % 100 == 0:
            active_suitcases = set()
            # 現在 match_history に残っている、アクティブなスーツケースIDを収集
            for matches in self.match_history.values():
                active_suitcases.update(matches.keys())

            # suitcase_match_history にあるが、現在の match_history にないIDを特定（非アクティブな荷物）
            inactive_suitcases = set(self.suitcase_match_history.keys()) - active_suitcases

            # 非アクティブなスーツケースの履歴を削除
            for suitcase_id in inactive_suitcases:
                del self.suitcase_match_history[suitcase_id]

    def _cleanup_first_approach_history(self):
        """
        初回接近履歴のタイムスタンプと関連情報を整理します。

        この関数は、一定期間（100フレーム）以上更新されていない初回接近の
        ペア情報（`approach_timestamps`）を削除し、それに伴い`first_approach_history`
        の無効なエントリもクリーンアップします。これにより、システムのメモリを
        節約し、古くなった情報がマッチング判断に影響を与えるのを防ぎます。

        Args:
            self (object): インスタンス自身。`approach_timestamps`と
                        `first_approach_history`にアクセスするために必要です。
        """
        current_frame = self._get_current_frame()
        expired_pairs = []


        # タイムスタンプをチェックし、100フレーム以上経過したペアを特定
        for (person_id, suitcase_id), timestamp in self.approach_timestamps.items():
            if current_frame - timestamp > 100:
                expired_pairs.append((person_id, suitcase_id))

        # 期限切れのタイムスタンプを削除
        for pair in expired_pairs:
            del self.approach_timestamps[pair]

        # タイムスタンプが削除された、無効な「最初の接近者」エントリを特定
        invalid_suitcases = []
        for suitcase_id, person_id in self.first_approach_history.items():
            if (person_id, suitcase_id) not in self.approach_timestamps:
                invalid_suitcases.append(suitcase_id)

        # first_approach_history から無効なエントリを削除
        for suitcase_id in invalid_suitcases:
            del self.first_approach_history[suitcase_id]

    def get_current_matches(self):
        """
        現在のフレームで確定したマッチングペアのリストを返します。

        この関数は、外部のコンポーネントが現在のマッチング結果にアクセスするための
        インターフェースとして機能します。返される辞書は、`update_matches`関数によって
        最終的に確定されたペアを反映します。

        Args:
            self (object): インスタンス自身。`confirmed_matches`にアクセスするために必要です。

        Returns:
            dict: 人物IDをキー、スーツケースIDを値とする、現在の確定マッチングのコピー。
        """
        # 外部からの変更を防ぐため、確定マッチングのコピーを返す
        return self.confirmed_matches.copy()

    def get_match_confidence(self, person_id):
        """
        特定の人物IDに対するマッチングの信頼度を取得します。

        この関数は、指定された人物IDが確定したマッチングペアの中に存在するか、
        またそのマッチング履歴がどれくらい強いかを判断します。
        ただし、現在の実装では、その人物が確定マッチングに含まれているか、
        または履歴が存在するかを単純にチェックし、信頼度スコアを返すロジックは
        実装されていません。常に0.0を返します。

        Args:
            person_id (str): 信頼度を問い合わせる人物のID。

        Returns:
            float: 信頼度スコア。現在のところは常に0.0を返します。
        """
        # 確定マッチングまたは履歴がない場合は、信頼度0.0を返す
        if person_id not in self.match_history or person_id not in self.confirmed_matches:
            return 0.0

        # 確定しているスーツケースIDとマッチングカウントを取得
        suitcase_id = self.confirmed_matches[person_id]
        match_count = self.match_history[person_id].get(suitcase_id, 0)

        # マッチングカウントを正規化し、信頼度スコアとして返す (最大値1.0)
        # スコアは 'min_match_frames * 2' を最大値の目安として計算される
        return min(1.0, match_count / (self.min_match_frames * 2))

    def get_matching_statistics(self):
        """マッチングアルゴリズムの統計レポートを生成します。

        この関数は、マッチングアルゴリズムの実行状態とパフォーマンスを評価するための
        複数の主要な指標を計算し、辞書として返します。これは、システムの健全性を監視し、
        デバッグやパフォーマンスチューニングに役立つ「ダッシュボード」として機能します。

        **算出される主要な統計:**
        1.  **total_persons_tracked**: アルゴリズムがこれまでに追跡した人物の総数。
        2.  **confirmed_matches**: 現在アクティブな、信頼性が高いと確定されたマッチングペアの数。
        3.  **average_confidence**: 確定されたマッチングペアの平均信頼度スコア。これにより、全体の
            マッチング品質を把握できます。
        4.  **conflicts_detected**: 一つのスーツケースが複数の人物にマッチングされている、
            論理的な矛盾（コンフリクト）の数。これは、アルゴリズムの堅牢性を示す重要な指標です。
        5.  **suitcase_usage**: 各スーツケースが確定マッチングにおいて何回使用されているかを示す情報。
        6.  **first_approach_records**: 初回接近履歴に記録されているスーツケースの数。
        7.  **active_approach_timestamps**: 追跡中の初回接近タイムスタンプの総数。

        Args:
            self (object): インスタンス自身。`match_history`、`confirmed_matches`などの内部状態に
                        アクセスするために必要です。

        Returns:
            dict: 上記の統計指標をキーと値のペアで格納した辞書。
        """
        # 追跡された人物の総数 (履歴ベース)
        total_persons_in_history = len(self.match_history)
        # 現在確定しているマッチングペアの数
        confirmed_matches_count = len(self.confirmed_matches)

        # 平均信頼度を計算
        avg_confidence = 0.0
        if confirmed_matches_count > 0:
            # 確定マッチング全体の信頼度スコアを合計
            total_confidence = sum(self.get_match_confidence(pid) for pid in self.confirmed_matches.keys())
            avg_confidence = total_confidence / confirmed_matches_count

        # スーツケースの使用状況を分析 (一つが複数人にマッチしていないかをチェック)
        suitcase_usage = {}
        for person_id, suitcase_id in self.confirmed_matches.items():
            suitcase_usage[suitcase_id] = suitcase_usage.get(suitcase_id, 0) + 1

        # 矛盾（一つのスーツケースに複数の人物が紐づいている状態）を検出
        conflicts = sum(1 for count in suitcase_usage.values() if count > 1)

        return {
            'total_persons_tracked': total_persons_in_history,
            'confirmed_matches': confirmed_matches_count,
            'average_confidence': avg_confidence,
            'conflicts_detected': conflicts,
            'suitcase_usage': suitcase_usage,
            'first_approach_records': len(self.first_approach_history),
            'active_approach_timestamps': len(self.approach_timestamps)
        }

    def get_gate_filter_statistics(self):
        """ゲートフィルタリングの統計情報を計算し取得します。

        この関数は、アルゴリズムの初期段階で行われるゲートエリアによるフィルタリングの
        有効性を評価するためのレポートを生成します。総試行回数、フィルタリングされた回数、
        およびフィルタリング率を計算します。これにより、フィルタリングがどれだけ多くの
        無関係なマッチング試行を効率的に除外しているかを把握できます。

        Args:
            self (object): インスタンス自身。`gate_filter_stats`辞書にアクセスするために必要です。

        Returns:
            dict: フィルタリングの統計情報を含む辞書。以下のキーが含まれます。
                - `total_attempts`: マッチングが試みられた総回数。
                - `area_filtered`: ゲートエリアの条件によってフィルタリングされた回数。
                - `total_filtered`: フィルタリングされた総回数。
                - `filter_rate`: フィルタリングされた割合（パーセント）。
        """
        stats = self.gate_filter_stats.copy()

        if stats['total_attempts'] > 0:
            # この実装では、ゲートエリアフィルタリング数のみを総フィルタリング数として計上
            stats['total_filtered'] = stats['area_filtered']
            # フィルタリング率を計算
            stats['filter_rate'] = stats['total_filtered'] / stats['total_attempts'] * 100
        else:
            stats['total_filtered'] = 0
            stats['filter_rate'] = 0.0

        return stats

    def print_gate_filter_report(self):
        """ゲートフィルタリングの統計レポートをコンソールに出力します。

        `get_gate_filter_statistics()`から取得した情報を用いて、ゲートフィルタリングの
        パフォーマンスを人間が読みやすい形式で表示します。このレポートは、システムの
        初期段階でのパフォーマンスをデバッグし、検証するのに役立ちます。
        """
        stats = self.get_gate_filter_statistics()

        print(f"\n🚪 ゲートフィルタリング統計レポート:")
        print(f"   総マッチング試行回数: {stats['total_attempts']}")
        print(f"   確定ゲートフィルタリング数: {stats['area_filtered']}")
        print(f"   総フィルタリング数: {stats['total_filtered']}")
        print(f"   フィルタリング率: {stats['filter_rate']:.1f}%")


class RealtimeMatchingStatistics:
    """
    リアルタイムのマッチング統計を収集し、レポートするクラスです。

    このクラスは、主要なデータ収集コンポーネントとマッチングアルゴリズムから
    現在のフレームデータを取得し、人物とスーツケースのマッチング状況に関する
    詳細な統計情報をリアルタイムで提供します。

    主な機能:
    - リアルタイムでの人物、スーツケース、およびマッチングペアの総数を報告。
    - マッチングされた人物、未マッチングの人物、および未マッチングの
      スーツケースの数を区別して表示。
    - マッチング率を計算し、アルゴリズムの効率性を評価する指標を提供。
    
    Args:
        data_collector (object): フレームデータを提供するデータ収集クラスのインスタンス。
        matcher (object): マッチングアルゴリズムのクラスインスタンス。
    """

    def __init__(self, data_collector, matcher):
        self.data_collector = data_collector
        self.matcher = matcher

    def get_realtime_statistics(self):
        """
        現在のフレームにおけるリアルタイムのマッチング統計を計算し、返します。

        このメソッドは、`data_collector`から最新のフレームデータを取得し、
        `matcher`クラスの`get_current_matches()`メソッドを用いて現在の
        確定マッチングペアを取得します。その後、これらの情報に基づいて、
        様々な統計指標を計算します。

        Returns:
            dict: 以下のキーを含む、現在のフレームの統計情報。
            - `current_persons`: 現在フレームにいる人物の総数。
            - `current_suitcases`: 現在フレームにいるスーツケースの総数。
            - `persons_with_suitcase`: スーツケースとマッチングした人物の数。
            - `persons_without_suitcase`: スーツケースとマッチングしていない人物の数。
            - `matched_suitcases`: マッチングしたスーツケースの総数。
            - `unmatched_suitcases`: マッチングしていないスーツケースの総数。
            - `matching_rate`: 人物に対するマッチング率（%）。
            - `confirmed_matches`: 確定されたマッチングペアの辞書。
        """
        current_frame = self.data_collector.get_current_frame_data()
        if not current_frame:
            return self._empty_stats()

        current_persons = [p['id'] for p in current_frame['persons']]
        current_suitcases = [s['id'] for s in current_frame['suitcases']]
        confirmed_matches = self.matcher.get_current_matches()

        persons_with_suitcase = []
        persons_without_suitcase = []

        for person_id in current_persons:
            if person_id in confirmed_matches:
                persons_with_suitcase.append(person_id)
            else:
                persons_without_suitcase.append(person_id)

        matched_suitcases = list(confirmed_matches.values())
        unmatched_suitcases = [s_id for s_id in current_suitcases if s_id not in matched_suitcases]

        total_persons = len(current_persons)
        match_rate = (len(persons_with_suitcase) / total_persons * 100) if total_persons > 0 else 0

        return {
            'current_persons': len(current_persons),
            'current_suitcases': len(current_suitcases),
            'persons_with_suitcase': len(persons_with_suitcase),
            'persons_without_suitcase': len(persons_without_suitcase),
            'matched_suitcases': len(matched_suitcases),
            'unmatched_suitcases': len(unmatched_suitcases),
            'matching_rate': match_rate,
            'confirmed_matches': confirmed_matches
        }

    def _empty_stats(self):
        """
        フレームデータが存在しない場合に、空の統計辞書を返します。
        
        Returns:
            dict: すべての統計値が0または空の辞書に設定された辞書。
        """
        return {
            'current_persons': 0, 'current_suitcases': 0,
            'persons_with_suitcase': 0, 'persons_without_suitcase': 0,
            'matched_suitcases': 0, 'unmatched_suitcases': 0,
            'matching_rate': 0, 'confirmed_matches': {}
        }


class MatcherBundle:
    """
    人物とスーツケースのマッチングシステム全体を統合し、管理するクラスです。

    このクラスは、複雑なマッチングアルゴリズムの複数のコンポーネント（データ収集、
    マッチングロジック、リアルタイム統計）を一つにまとめ、シンプルで使いやすい
    公開インターフェースを提供します。これにより、外部のアプリケーションは
    内部の詳細を知ることなく、高レベルなマッチング処理を実行できます。

    主要な役割は「**オーケストレーション**（orchestration）」です。
    データ収集からマッチング、そしてレポート生成までのエンドツーエンドの
    プロセスを調整・実行します。

    入力：
      - persons (np.ndarray[Np,5]): 検出された人物のトラッキングデータ。
      - suitcases (np.ndarray[Ns,5]): 検出されたスーツケースのトラッキングデータ。
      - keypoints (List[np.ndarray(17,3)]): 人物の姿勢キーポイントデータ。

    出力：
      - current: 現在フレームでの最適なマッチングペア。
      - confirmed: 長期的に確定されたマッチングペア。
      - stats: リアルタイム、マッチング、ゲートフィルタリングの各統計レポート。
    """
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        distance_threshold: int = 200,
        overlap_threshold: float = 0.1,
        min_match_frames: int = 1,
        max_frames_history: int = 30,
    ):
        """
        MatcherBundleのコンストラクタ。

        必要なすべてのサブコンポーネントをインスタンス化し、システムを初期化します。

        Args:
            frame_width (int): 処理するビデオフレームの幅。
            frame_height (int): 処理するビデオフレームの高さ。
            distance_threshold (int, optional): マッチングの基本距離しきい値。デフォルトは200。
            overlap_threshold (float, optional): BBoxの重なり合いしきい値。デフォルトは0.1。
            min_match_frames (int, optional): マッチングを確定するために必要な最小連続フレーム数。デフォルトは1。
            max_frames_history (int, optional): データコレクターが保持する履歴フレームの最大数。デフォルトは30。
        """
        # データ収集器を初期化（トラッキングデータの履歴を保持）
        self.collector = TrackingDataCollector(max_frames_history=max_frames_history)
        # コアとなるマッチングアルゴリズムを初期化（複雑なマッチングロジックを担当）
        self.matcher = ImprovedRealtimePersonSuitcaseMatcher(
            distance_threshold=distance_threshold,
            overlap_threshold=overlap_threshold,
            min_match_frames=min_match_frames,
            frame_width=frame_width,
            frame_height=frame_height
        )

        # リアルタイム統計レポート生成クラスを初期化（可視化やデバッグ用）
        self.rt_stats = RealtimeMatchingStatistics(self.collector, self.matcher)


    def step(
        self,
        frame_no: int,
        persons: np.ndarray,
        suitcases: np.ndarray,
        keypoints_list: Optional[List[np.ndarray]] = None,
        item_types: Optional[List[str]] = None  # 追加のパラメータ
    ) -> Dict[str, Any]:
        """
        システムのメイン処理ループ。毎フレームこのメソッドが呼び出されます。

        与えられたフレームデータを内部コレクターに追加し、マッチングアルゴリズムを
        実行し、最終的なマッチング結果と各種統計情報を返します。

        Args:
            frame_no (int): 現在のフレーム番号。
            persons (np.ndarray): 現在フレームで検出された人物のトラッキングデータ。
            suitcases (np.ndarray): 現在フレームで検出されたスーツケースのトラッキングデータ。
            keypoints_list (Optional[List[np.ndarray]]): 各人物の姿勢キーポイントのリスト。デフォルトはNone。
            item_types (Optional[List[str]]): 各スーツケースのアイテムタイプ。デフォルトはNone。

        Returns:
            Dict[str, Any]: 以下のキーを持つ、処理結果の辞書。
                            - 'current': 現在フレームで解決された最適なマッチングペア。
                            - 'confirmed': 長期的に確定されたマッチングペア。
                            - 'stats': リアルタイム、マッチング、ゲートフィルタリングの各統計レポート。
        """
        keypoints_list = keypoints_list or []
        item_types = item_types or []  # デフォルト値の追加処理

        # 1. データをコレクターに追加（履歴保持と処理のための準備）
        self.collector.add_frame_data(
            frame_number=frame_no,
            person_tracks=persons if persons is not None else np.empty((0, 5)),
            suitcase_tracks=suitcases if suitcases is not None else np.empty((0, 5)),
            raw_keypoints=keypoints_list,
            item_types=item_types  # 追加で渡す
        )

        # 2. コレクターから最新データを取得
        cur = self.collector.get_current_frame_data()

        # 3. マッチングアルゴリズムを実行し、現在のマッチングを更新（コア処理）
        current_frame_matches = self.matcher.update_matches(cur)

        # 4. 長期的に確定したマッチング結果を取得
        confirmed = self.matcher.get_current_matches()

        # 5. 結果と統計情報を統合して返す
        return {
            "current": current_frame_matches,
            "confirmed": confirmed,
            "stats": {
                "realtime": self.rt_stats.get_realtime_statistics(),
                "matching": self.matcher.get_matching_statistics(),
                "gate": self.matcher.get_gate_filter_statistics()
            }
        }

    def finalize(self) -> Dict[str, Any]:
        """
        マッチング処理の最終結果を返します。

        通常、ビデオストリームの処理終了時に一度だけ呼び出され、最終的な確定マッチングと
        システム全体の統計情報を提供します。

        Returns:
            Dict[str, Any]: 最終的な確定マッチングと、終了時点での統計情報。
        """
        return {
            "confirmed": self.matcher.get_current_matches(),
            "realtime": self.rt_stats.get_realtime_statistics(),
            "matching": self.matcher.get_matching_statistics(),
            "gate": self.matcher.get_gate_filter_statistics()
        }
