# module/model_loader.py
from ultralytics import YOLO

_model_cache = {}

def load_model(model_name: str):
    """
    指定された名前の YOLO モデルをロードする関数。

    本関数は、与えられた `model_name` をもとに Ultralytics YOLO モデルを読み込みます。
    読み込み済みのモデルは内部キャッシュ (`_model_cache`) に保持され、同じ名前での再呼び出し時には
    ディスク読み込みをスキップしてキャッシュから即座に返却されます。
    `.pt` 拡張子が付いていない場合は、自動的に候補として付加したファイル名も試行します。

    引数:
        model_name (str):
            読み込み対象のモデル名またはファイルパス。
            例: `"yolov11n-pose"`, `"runs/train/exp/weights/best"`, `"custom_model.pt"`。

    戻り値:
        YOLO:
            Ultralytics YOLO モデルオブジェクト。
            ロード成功時に返され、以降は `_model_cache` に保存される。

    例外:
        RuntimeError:
            すべての候補（model_name, model_name+".pt"）でロードに失敗した場合に発生。

    動作仕様:
        1. `model_name` を候補とし、拡張子 `.pt` が無ければ `model_name+".pt"` も候補に追加。
        2. 候補名がキャッシュに存在する場合はキャッシュから返却。
        3. Ultralytics の `YOLO` クラスを使ってロードを試み、成功したらキャッシュに保存。
        4. 全候補が失敗した場合は `RuntimeError` を送出。

    使用例:
        ```python
        from module.model_loader import load_model

        # 標準モデルをロード
        model = load_model("yolov11n-pose")

        # キャッシュから再利用（2回目以降は高速）
        model2 = load_model("yolov11n-pose")
        ```
    """
    candidates = [model_name]
    if not model_name.endswith(".pt"):
        candidates.append(model_name + ".pt")

    # 最初にキャッシュをチェック
    for key in candidates:
        if key in _model_cache:
            print(f"Model '{key}' loaded from cache.")
            return _model_cache[key]

    last_err = None
    # キャッシュにない場合、候補名を一つずつ試してロード
    for key in candidates:
        try:
            print(f"Loading model: {key}")
            # YOLOモデルのインスタンス化を試みる
            model = YOLO(key)
            # ロード成功時にキャッシュに保存
            _model_cache[key] = model
            print("Model loaded successfully.")
            return model
        except Exception as e:
            last_err = e
            # 失敗した場合は次の候補を試す
            continue
        
    # すべての候補でロードに失敗した場合、エラーを送出
    raise RuntimeError(f"Failed to load model with names {candidates}: {last_err}")
