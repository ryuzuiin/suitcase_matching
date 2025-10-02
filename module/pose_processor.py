# module/pose_processor.py
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional
import numpy as np

@dataclass
class BBoxResult:
    """
    バウンディングボックス検出結果を格納するためのデータクラス。

    このクラスは、物体検出アルゴリズムによって生成される単一の
    バウンディングボックスの情報を、構造化された形式で保持します。

    Args:
        bbox (List[float]): 物体のバウンディングボックス座標。
                            通常、`[x1, y1, x2, y2]` の形式。
        method (str): バウンディングボックスが検出された手法またはモデルの名前。
        confidence (float): 検出の信頼度スコア。通常、0.0から1.0の範囲。
        metadata (Dict): 検出に関する追加情報（例：クラスID、追跡ID、
                         属性情報など）を格納する辞書。
    """
    bbox: List[float]
    method: str
    confidence: float
    metadata: Dict  # extra information

class ConversionMethod(Enum):
    """
    キーポイントからバウンディングボックスへの変換メソッドの列挙型

    Methods:
        SIMPLE_ENCLOSING: 全ての有効なキーポイントを囲む最小矩形を生成
                         最も基本的で高速な方法
        REGIONAL_PRIORITY: 身体部位の優先度に基づいてボックスを生成
                          頭部や胴体を優先し、より安定した検出を実現
    """
    SIMPLE_ENCLOSING = "simple_enclosing"
    REGIONAL_PRIORITY = "regional_priority"

class KeypointUtils:
    """
    人体キーポイントの処理とバウンディングボックス変換のユーティリティクラス

    COCO形式の17個のキーポイントを処理し、様々な身体部位の
    グループ化と変換機能を提供します。
    """

    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    """
    キーポイント名のリスト（COCO形式、17点）
    インデックス対応:
        0: 鼻
        1-2: 左目、右目
        3-4: 左耳、右耳
        5-6: 左肩、右肩
        7-8: 左肘、右肘
        9-10: 左手首、右手首
        11-12: 左腰、右腰
        13-14: 左膝、右膝
        15-16: 左足首、右足首
    """

    BODY_REGIONS = {
        'head': [0, 1, 2, 3, 4],
        'torso': [5, 6, 11, 12],
        'arms': [7, 8, 9, 10],
        'legs': [13, 14, 15, 16],
        'upper_body': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'lower_body': [11, 12, 13, 14, 15, 16],
        'core': [0, 5, 6, 11, 12]
    }
    """
    身体部位ごとのキーポイントインデックスマッピング

    部位説明:
        head: 頭部（鼻、目、耳）
        torso: 胴体（肩、腰）
        arms: 腕部（肘、手首）
        legs: 脚部（膝、足首）
        upper_body: 上半身（頭部＋胴体上部＋腕）
        lower_body: 下半身（腰＋脚）
        core: コア部分（鼻＋肩＋腰、最も安定した点）
    """

    @staticmethod
    def validate_keypoints(keypoints: np.ndarray) -> bool:
        """
        キーポイント配列の形式を検証

        Args:
            keypoints (np.ndarray): 検証対象のキーポイント配列

        Returns:
            bool: 有効な形式の場合True、無効な場合False

        検証内容:
            - NumPy配列であること
            - 形状が(17, 3)または(17, 2)であること
            - (17, 3): [x, y, confidence]形式
            - (17, 2): [x, y]形式（信頼度なし）
        """
        if not isinstance(keypoints, np.ndarray):
            return False # NumPy配列でない場合は無効
        if keypoints.shape != (17, 3) and keypoints.shape != (17, 2):
            return False # (17, 3) または (17, 2) の形状でない場合は無効
        return True

    @staticmethod
    def get_valid_keypoints(keypoints: np.ndarray, confidence_threshold: float = 0.3) -> np.ndarray:
        """
        信頼度閾値を超えるキーポイントのみを抽出

        Args:
            keypoints (np.ndarray): 入力キーポイント配列 (17, 3)または(17, 2)
            confidence_threshold (float): 信頼度の閾値（デフォルト: 0.3）
                                         0.0〜1.0の範囲で指定

        Returns:
            np.ndarray: 有効なキーポイントのみを含む配列

        処理:
            - 3列の場合: 3列目の信頼度が閾値を超える点を抽出
            - 2列の場合: 座標が(0, 0)でない点を有効とみなす
        """
        if keypoints.shape[1] == 3:
            # 信頼度（3列目）が閾値を超えるキーポイントを抽出
            valid_mask = keypoints[:, 2] > confidence_threshold
            return keypoints[valid_mask]
        else:
            # 信頼度がない場合、座標(x, y)が両方とも0でない点を有効とみなす
            valid_mask = (keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)
            return keypoints[valid_mask]

    @staticmethod
    def get_region_keypoints(keypoints: np.ndarray, region: str) -> np.ndarray:
        """
        指定された身体部位のキーポイントを取得

        Args:
            keypoints (np.ndarray): 全キーポイント配列 (17, 3)または(17, 2)
            region (str): 身体部位名
                         選択可能: 'head', 'torso', 'arms', 'legs',
                                  'upper_body', 'lower_body', 'core'

        Returns:
            np.ndarray: 指定部位のキーポイントのみを含む配列

        Raises:
            ValueError: 未知の部位名が指定された場合

        注意:
            コード内にタイポがあります: KeypointUtlis → KeypointUtils
        """
        if region not in KeypointUtils.BODY_REGIONS:
            raise ValueError(f"Unknown region: {region}")  # 定義されていない部位名の場合はエラー

        indices = KeypointUtils.BODY_REGIONS[region]  # 部位に対応するインデックスを取得
        return keypoints[indices]  # インデックスを使ってキーポイントを抽出


class KeypointsToBBoxConverter:
    """
    キーポイント座標からバウンディングボックスへの変換を行うクラス

    複数の変換アルゴリズムを提供し、シーンや検出精度に応じて
    最適な方法を選択できる柔軟な変換システムを実装します。
    """

    def __init__(self):
        """
        変換器の初期化

        属性:
            default_confidence_threshold (float): デフォルトの信頼度閾値（0.3）
            history_buffer (list): 時系列安定化用の履歴バッファ（将来の拡張用）
            max_history_length (int): 履歴バッファの最大長（5フレーム）
        """
        # デフォルトの信頼度閾値を設定
        self.default_confidence_threshold = 0.3


    def convert(self, keypoints: np.ndarray, method: ConversionMethod,
                confidence_threshold: float = None, **kwargs) -> Optional[BBoxResult]:
        """
        キーポイントからバウンディングボックスへの変換メインメソッド

        Args:
            keypoints (np.ndarray): 入力キーポイント配列 (17, 3)または(17, 2)
                                   形式: [[x, y, confidence], ...]
            method (ConversionMethod): 使用する変換メソッド
                                      SIMPLE_ENCLOSING または REGIONAL_PRIORITY
            confidence_threshold (float, optional): 信頼度閾値（None時はデフォルト値0.3を使用）
            **kwargs: メソッド固有のパラメータ
                     - margin: 包囲ボックスのマージン（ピクセル）
                     - region_priority: 優先する身体部位のリスト

        Returns:
            Optional[BBoxResult]: 変換結果を含むBBoxResultオブジェクト
                                 変換失敗時はNoneを返す

        Raises:
            ValueError: 未知の変換メソッドが指定された場合
        """
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold  # 閾値が指定されていなければデフォルト値を使用

        if not KeypointUtils.validate_keypoints(keypoints):
            return None  # キーポイント形式が無効な場合は処理をスキップ

        method_map = {
            ConversionMethod.SIMPLE_ENCLOSING: self._simple_enclosing,
            ConversionMethod.REGIONAL_PRIORITY: self._regional_priority
        }

        if method not in method_map:
            raise ValueError(f"Unknown conversion method: {method}")

        return method_map[method](keypoints, confidence_threshold, **kwargs)

    def _simple_enclosing(self, keypoints: np.ndarray, confidence_threshold: float,
                          **kwargs) -> Optional[BBoxResult]:
        """
        シンプル包囲法によるバウンディングボックス生成（内部メソッド）

        Args:
            keypoints (np.ndarray): キーポイント配列
            confidence_threshold (float): 信頼度閾値
            **kwargs: 追加パラメータ
                     - margin (int): ボックス周囲のマージン（デフォルト: 15ピクセル）

        Returns:
            Optional[BBoxResult]: バウンディングボックス結果
                                 有効なキーポイントが2個未満の場合はNone

        処理アルゴリズム:
            1. 信頼度閾値を超えるキーポイントを抽出
            2. 全有効点を囲む最小矩形を計算
            3. 指定マージンを追加
            4. 平均信頼度を計算
            5. メタデータと共に結果を返却
        """
        # 1. 信頼度閾値を超えるキーポイントを抽出
        valid_keypoints = KeypointUtils.get_valid_keypoints(keypoints, confidence_threshold)

        if len(valid_keypoints) < 2:
            return None

        coords = valid_keypoints[:, :2]

        # 2. 全有効点を囲む最小矩形を計算
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)

        # 3. 指定マージンを追加
        margin = kwargs.get('margin', 15)
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = x_max + margin
        y_max = y_max + margin

        # 4. 平均信頼度を計算
        if keypoints.shape[1] == 3:
            avg_confidence = np.mean(valid_keypoints[:, 2])
        else:
            avg_confidence = 0.8

        bbox = [float(x_min), float(y_min), float(x_max), float(y_max), float(avg_confidence)]

        metadata = {
            'valid_keypoints_count': len(valid_keypoints),
            'total_keypoints': len(keypoints),
            'margin_used': margin
        }

        print(f"Simple enclosing bbox: {bbox} with length={len(bbox)}")

        return BBoxResult(bbox, "simple_enclosing", avg_confidence, metadata)

    def _regional_priority(self, keypoints: np.ndarray, confidence_threshold: float,
                            **kwargs) -> Optional[BBoxResult]:
        """
        身体部位優先法によるバウンディングボックス生成（内部メソッド）

        Args:
            keypoints (np.ndarray): キーポイント配列
            confidence_threshold (float): 信頼度閾値
            **kwargs: 追加パラメータ
                     - region_priority (list): 優先する身体部位のリスト
                                             デフォルト: ['core', 'upper_body', 'head', 'torso']

        Returns:
            Optional[BBoxResult]: バウンディングボックス結果
                                 全部位で検出失敗の場合はシンプル包囲法にフォールバック

        処理アルゴリズム:
            1. 優先度順に身体部位をチェック
            2. 各部位で有効なキーポイントが2個以上あるか確認
            3. 部位に応じた専用マージンを適用
               - head: x=25, y=30 ピクセル
               - core: x=30, y=50 ピクセル
               - upper_body: x=25, y=40 ピクセル
               - torso: x=20, y=35 ピクセル
            4. 特定部位（head, upper_body）では推定による拡張を実施
               - head: 高さを4倍に拡張（全身推定）
               - upper_body: 高さを1.5倍に拡張
            5. 最初に条件を満たした部位でボックスを生成

        フォールバック:
            全部位で検出失敗の場合は_simple_enclosingメソッドを呼び出し
        """
        # 優先度リストを取得（デフォルト: core, upper_body, head, torso）
        region_priority = kwargs.get('region_priority', ['core', 'upper_body', 'head', 'torso'])

        for region in region_priority:  # 優先度順に部位をチェック
            try:
                region_keypoints_indices = KeypointUtils.BODY_REGIONS[region]
                region_keypoints = keypoints[region_keypoints_indices]
                valid_region_keypoints = KeypointUtils.get_valid_keypoints(
                    region_keypoints, confidence_threshold)

                if len(valid_region_keypoints) >= 2:  # 有効なキーポイントが2個以上ある場合
                    coords = valid_region_keypoints[:, :2]

                    x_min, y_min = np.min(coords, axis=0)
                    x_max, y_max = np.max(coords, axis=0)

                    # 部位別の専用マージン設定
                    region_margins = {
                        'head': {'x': 25, 'y': 30},
                        'core': {'x': 30, 'y': 50},
                        'upper_body': {'x': 25, 'y': 40},
                        'torso': {'x': 20, 'y': 35}
                    }

                    margins = region_margins.get(region, {'x': 20, 'y': 30})
                    x_min = x_min - margins['x']
                    y_min = y_min - margins['y']
                    x_max = x_max + margins['x']
                    y_max = y_max + margins['y']

                    # 特定部位での推定拡張
                    if region in ['head', 'upper_body']:
                        height = y_max - y_min
                        if region == 'head':
                            y_max = y_max + height * 4  # 頭部をベースに高さを4倍に拡張（全身推定）
                        elif region == 'upper_body':
                            y_max = y_max + height * 0.5  # 上半身をベースに高さを1.5倍に拡張（下半身を推定）


                    if keypoints.shape[1] == 3:
                        avg_confidence = np.mean(valid_region_keypoints[:, 2])
                    else:
                        avg_confidence = 0.8

                    bbox = [x_min, y_min, x_max, y_max, avg_confidence]

                    metadata = {
                        'region_used': region,
                        'region_keypoints_count': len(valid_region_keypoints),
                        'margins': margins
                    }

                    print(f"Region priority bbox: {bbox} with length={len(bbox)}")
                    return BBoxResult(bbox, "regional_priority", avg_confidence, metadata)

            except (KeyError, IndexError):
                continue  # 該当部位のインデックスエラーなどが発生した場合は次の部位へ

        # 全部位で失敗した場合はシンプル包囲法にフォールバック
        return self._simple_enclosing(keypoints, confidence_threshold, **kwargs)
    
def batch_convert_keypoints(keypoints_list: List[np.ndarray],
                            converter: Optional[KeypointsToBBoxConverter] = None,
                            method: ConversionMethod = ConversionMethod.REGIONAL_PRIORITY,
                            confidence_threshold: Optional[float] = None,
                            **kwargs) -> np.ndarray:
    
    """
    複数人物のキーポイント配列をまとめてバウンディングボックスへ変換する関数。

    本関数は、人物ごとに検出された 17 キーポイントの配列を入力として受け取り、
    指定された変換メソッド（`SIMPLE_ENCLOSING` または `REGIONAL_PRIORITY`）を用いて
    バウンディングボックス `[x1, y1, x2, y2, confidence]` へ一括変換します。
    成功した変換のみを配列にまとめ、最終的に `(N, 5)` の NumPy 配列として返却します。

    引数:
        keypoints_list (List[np.ndarray]):
            複数人物分のキーポイント配列リスト。
            各要素は `(17, 3)` または `(17, 2)` 形式で、[x, y, confidence] または [x, y] を含む。
        converter (Optional[KeypointsToBBoxConverter]):
            変換を実行するコンバータオブジェクト。
            None の場合は内部で新規に `KeypointsToBBoxConverter` を生成。
        method (ConversionMethod):
            使用する変換方式。
            - `ConversionMethod.SIMPLE_ENCLOSING`: 全有効点を囲む最小矩形で生成
            - `ConversionMethod.REGIONAL_PRIORITY`: 身体部位の優先度に基づき安定した矩形を生成
        confidence_threshold (Optional[float]):
            信頼度閾値。None の場合はコンバータのデフォルト値（0.3）が使用される。
        **kwargs:
            メソッド固有の追加パラメータ。
            - margin (int): SIMPLE_ENCLOSING の場合、矩形周囲に付与するマージン（ピクセル）
            - region_priority (list[str]): REGIONAL_PRIORITY の場合、優先度付きで処理する部位名リスト
                                           例: ['core', 'upper_body', 'head', 'torso']

    戻り値:
        np.ndarray:
            形状 `(N, 5)` の NumPy 配列。
            各行は `[x1, y1, x2, y2, confidence]` を表す。
            変換に失敗した場合や有効な検出がなかった場合は空配列 `(0, 5)` を返す。

    出力ログ:
        - 変換対象人数と使用メソッドを標準出力に表示。
        - REGIONAL_PRIORITY の場合、優先度リストを表示。
        - 各人物ごとの処理結果（成功／失敗、および使用部位）を表示。
        - 最終的に成功件数を表示。

    使用例:
        ```python
        from module.pose_processor import batch_convert_keypoints, ConversionMethod

        keypoints_list = [np.random.rand(17, 3), np.random.rand(17, 3)]
        detections = batch_convert_keypoints(
            keypoints_list,
            method=ConversionMethod.REGIONAL_PRIORITY,
            confidence_threshold=0.2
        )
        print(detections.shape)  # => (人数, 5)
        ```

    注意:
        - 各人物で 2 点以上の有効キーポイントがなければ変換は失敗する。
        - REGIONAL_PRIORITY モードでは、優先度リストの順に部位をチェックし、
          最初に条件を満たした部位で矩形を生成する。
        - すべての人物で失敗した場合は空配列を返却する。
    """
    if converter is None:
        converter = KeypointsToBBoxConverter()

    detections = []

    print(f"Converting {len(keypoints_list)} keypoint sets using {method.name}...")

    if method == ConversionMethod.REGIONAL_PRIORITY:
        region_priority = kwargs.get('region_priority', ['core', 'upper_body', 'head', 'torso'])
        print(f"  Region priority: {region_priority}")  # 部位優先法の場合、使用する優先度リストを出力


    for i, keypoints in enumerate(keypoints_list):
        result = converter.convert(
            keypoints,
            method,
            confidence_threshold,
            **kwargs
        )  # 各人物のキーポイントを変換

        if result:
            detections.append(result.bbox)  # 成功したBBoxをリストに追加
            if method == ConversionMethod.REGIONAL_PRIORITY and 'region_used' in result.metadata:
                print(f"  Person {i+1}: Success (used {result.metadata['region_used']} region)")  # 部位優先法の場合、使用部位を出力
            else:
                print(f"  Person {i+1}: Success")
        else:
            print(f"  Person {i+1}: Failed")  # 変換失敗を出力

    if detections:
        # 成功したBBoxリストをNumPy配列 (N, 5) に変換して返却
        detections_array = np.asarray(detections, dtype=np.float32).reshape(-1, 5)
        print(f"Total successful conversions: {len(detections)}")
        return  detections_array
    else:
        print("No successful conversions.")
        return np.empty((0, 5), dtype=np.float32)# 成功がなければ空の (0, 5) 配列を返却

