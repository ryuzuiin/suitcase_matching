# スーツケース所持者の検知プログラム
 - 検知したい動画を読み込むことで、人とスーツケースの検知、IDの付与、人とスーツケースの結び付けが出来るプログラムとなっている。


## ■実行環境
 - PGM実行時の環境：Google Colab
 - FineTuningの際の実行環境：GeForce RTX 3060

## 実行前の書き換え箇所
 - 実行用pgm.ipynb
   - base_dir
 - config
   - BASE_DIR

## ■フォルダ構成
```
.
├── README.md
├── config
│   └── config.py             --------- 設定用ファイル
├── libs
│   └── preprocessing.py      --------- 動画の前処理用ファイル
├── module
│   ├── detectors.py          --------- 物体検出用ファイル
│   ├── draw_utils.py         --------- 可視化用ファイル
│   ├── frame_processor.py    --------- 単一フレームの処理用ファイル
│   ├── matching.py           --------- 人物×荷物マッチング用ファイル
│   ├── model_loader.py       --------- 学習済みモデルの読込用ファイル
│   ├── pipeline_dual.py      --------- 人物と荷物を同時に検出・追跡しマッチングと可視化を行う用ファイル
│   ├── pose_processor.py     --------- キーポイントから人物バウンディングボックスを生成する用ファイル
│   └── sort_tracker.py       --------- SORTによる多対象追跡用ファイルファイル
└── pgm
    └── 実行用pgm.ipynb        --------- 実行用プログラム

```


## ■ファイル説明
### config
- `config.py` : 全体設定値を定義する用ファイル
```
BASE_DIR: フォルダ構成を作成するためのベースとなるパス。
         「.」の部分にあたる場所を指す。

INPUT_PATH: 入力動画ファイルのパス。  

OUTPUT_PATH: 出力用ディレクトリのパス。  
        　   `POSE_MODEL_NAME` の値をフォルダ名に付与し、  `{モデル名}_output` という形式で BASE_DIR 配下に作成される。

OUTPUT_VIDEO_PATH: 出力動画ファイルの保存先パス。  
        　         ファイル名は、`output_{POSE_MODEL_NAME}_{TARGET_FPS}fps.mp4` という形式になり、
                　 使用モデル名（POSE_MODEL_NAME）と目標フレームレート（TARGET_FPS）が含まれる。

POSE_MODEL_NAME: 姿勢推定モデルの名前。人の骨格（ポーズ）を検出・解析するために使用するモデル。
	         ex.) "yolo11m-pose"   もしくは "yolo11m-pose.pt"

SUITCASE_MODEL_NAME: 物体の検出モデルの名前。「スーツケース」のような特定の物体を検出するために使用するモデル。 
		　　　ex.) "yolo11x"    もしくは "yolo11x.pt"

TARGET_FPS: 目標フレームレート。
            処理後の動画や解析の対象とする、1秒あたりのコマ数。

TEST_DURATION_SECONDS: 動画の処理時間（秒）。（但し、0秒起点） 
```

### libs
- `preprocessing.py` : 動画の入出力管理とフレーム間引きなど前処理を行う用ファイル

### module
- `detectors.py` : YOLOで物体検出を行う用ファイル  
- `draw_utils.py` : バウンディングボックスやIDをフレーム上に描画する用ファイル  
- `frame_processor.py` : 単一フレームで推論→変換→追跡→描画まで行う処理クラス用ファイル  
- `matching.py` : 人物と荷物のマッチング処理（距離・姿勢・履歴で評価し統計管理）用ファイル  
- `model_loader.py` : YOLOモデルの読み込みとキャッシュ再利用を行う用ファイル   
- `pipeline_dual.py` : 人物と荷物を同時に検出・追跡しマッチングと可視化を行う統合処理用ファイル
- `pose_processor.py` : キーポイントから人物バウンディングボックスを生成する用ファイル  
- `sort_tracker.py` : SORTによる多対象追跡（Kalman＋ハンガリアンでID付与・更新）用ファイル

### pgm

- `実行用pgm.ipynb` : 全体の処理パイプラインを実行するためのGoogle Colaboratory Notebook
