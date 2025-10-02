import os, time 
from config.config import output_dir, input_path, output_video_path, target_fps, model_name, test_duration_seconds 
from libs.preprocessing import VideoProcessor 
from module.model_loader import load_model 
from module.frame_processor import FrameProcessor 
from module.pipeline_dual import DualPipeline 
from module.pose_processor import ConversionMethod

def main():
    """
    指定の設定値に基づき、動画を読み込み、人物のみ/人物+荷物（デュアル）いずれかの
    パイプラインで処理し、可視化したフレームを動画として出力するメイン関数。

    実施内容（実行レベル・利他型）:
        - 各モジュールの役割が一目で分かるよう、処理内容・入出力・フローを明確化した構成で実装。
        - レビュー/改修/拡張時に参照しやすい形（引数なし・外部設定駆動・例外処理/解放の徹底）で整備。
        - 「person」「dual」モード切替により、用途に応じて負荷と機能を選択可能。

    処理フロー:
        1) 出力ディレクトリ作成（存在しない場合）
        2) モード選択（"person" or "dual"）
           - person: YOLO-Pose → FrameProcessor（SORT追跡/描画）
           - dual  : YOLO-Pose + YOLO(det) → DualPipeline（人物/荷物の統合追跡・可視化・統計）
        3) モデル読み込み
           - pose_model = load_model(model_name)
           - suitcase_model = load_model("yolo11x")（dual時のみ）
        4) VideoProcessor の初期化と開始（入出力パス/目標FPS/フレーム間引き）
        5) 目標フレーム数までループ
           - フレーム読み込み
           - frame_interval に従い「完全処理」するか判定
           - person: FrameProcessor.process_frame()
           - dual  : DualPipeline.process()
           - 書き出し（annotated_frame）
        6) 終了ログ・dual時のサマリ出力（現在統計/確定マッチ）
        7) 例外処理と最終的なリソース解放

    モード仕様:
        - mode="person":
            ・人物のみの追跡を実施（pose → bbox変換 → SORT → 可視化）
            ・FrameProcessor パラメータ例: sort_max_age=50, sort_min_hits=2, iou_threshold=0.3
        - mode="dual":
            ・人物 + 荷物（スーツケース等）の統合追跡を実施
            ・DualPipeline を用い、下記を初期化:
                pose_model            : YOLO-Pose
                suitcase_model        : 荷物用 YOLO（例: yolo11x）
                frame_width/height    : VideoProcessor から取得
                conversion_method     : ConversionMethod.REGIONAL_PRIORITY
                conversion_kwargs     : {'region_priority': ['core'], 'confidence_threshold': 0.4}
                suitcase_confidence   : 0.40（COCO class_id=28: suitcase）
                suitcase_class_id     : 28
                person_sort           : (50, 2, 0.3)
                suitcase_sort         : (20, 1, 0.2)
                match_dist            : 250（ピクセル距離）
                match_overlap         : 0.1（IoUしきい値）
                match_min_frames      : 1（確定マッチ最小連続フレーム）

    入出力:
        引数: なし（すべて config/config.py の設定を参照）
        出力: 可視化済みの動画ファイル（processor.output_path / config.output_video_path）
        標準出力ログ:
            - 開始/終了、処理フレーム数、フレーム間引き間隔
            - 入力の終端検出
            - dualモード時のサマリ（現在人数/荷物合計/確定マッチ統計）

    設定（config/config.py から読み込み）:
        - input_path, output_video_path, target_fps, model_name, output_dir, test_duration_seconds
        - VideoProcessor.frame_interval により「完全処理の頻度」を制御
        - total_frame_to_process = int(target_fps * test_duration_seconds * 3)
          （テスト用に拡張係数を掛けている点に留意）

    チューニング観点（改修者向けの参照ポイント）:
        - 推論負荷と精度のバランス: frame_interval / model の軽量化/信頼度閾値
        - マッチング安定性: person_sort / suitcase_sort / match_dist / match_overlap / match_min_frames
        - 人物bboxの安定化: conversion_method と conversion_kwargs（region_priority の選択）

    例外/リソース管理:
        - IOError/Exception を捕捉してログ出力（Exception は再送出）
        - finally ブロックで VideoProcessor のリソースを必ず解放

    使用例:
        ```bash
        python -m module.main_processor
        ```
    """
    # 必要モジュール: os, time の import を忘れずに
    os.makedirs(output_dir, exist_ok=True)

    mode = "dual"  # "person" or "dual"

    # モデル読み込み（pose: yolo11*-pose / 荷物: 通常の検出モデル）
    pose_model = load_model(model_name)                  # 例: "yolo11m-pose"
    suitcase_model = load_model("yolo11x") if mode == "dual" else None

    processor = VideoProcessor(input_path, output_video_path, target_fps)
    try:
        print("Starting video processing...")
        processor.start_processing()

        if mode == "person":
            # 人物のみ：既存のフレームプロセッサ
            frame_processor = FrameProcessor(
                model=pose_model, sort_max_age=50, sort_min_hits=2, iou_threshold=0.3
            )
        else:
            # デュアル（人 + 荷物）：三種類の荷物を統合追跡
            pipeline = DualPipeline(
                pose_model=pose_model,
                suitcase_model=suitcase_model,
                frame_width=processor.width,
                frame_height=processor.height,
                # キーポイント→bbox 変換の戦略
                conversion_method=ConversionMethod.REGIONAL_PRIORITY,
                conversion_kwargs={'region_priority': ['core'], 'confidence_threshold': 0.4},
                # 荷物検出の信頼度（任意調整：handbag はやや低め、suitcase は標準）
                suitcase_confidence=0.40,    # 28: suitcase
                # backpack_confidence=0.40,    # 24: backpack
                # handbag_confidence=0.35,     # 26: handbag
                # COCO クラスID（デフォルトのままでもOK）
                # backpack_class_id=24,
                # handbag_class_id=26,
                suitcase_class_id=28,
                # SORT/マッチングのパラメータ（必要に応じて調整）
                person_sort=(50, 2, 0.3),    # 人側は多少厳しめの安定化
                suitcase_sort=(20, 1, 0.2),       # 荷物側はデフォルトのまま
                match_dist=250,              # ピクセル閾値（後で正規化版に置換も可）
                match_overlap=0.1,
                match_min_frames=1,
            )

        frame_count = 0
        # 処理フレーム数（テスト用）。必要に応じて調整。
        total_frame_to_process = int(processor.target_fps * test_duration_seconds * 3)

        print(f"Target: {total_frame_to_process} frames, process every {processor.frame_interval} frames")
        start_time = time.time()

        while frame_count < total_frame_to_process:
            ret, frame = processor.read_frame()
            if not ret:
                print("End of video stream.")
                break

            # フレーム間引き（VideoProcessor 側の frame_interval を利用)
            # frame = frame[321:, 250:1600]
            is_full_process = (frame_count % processor.frame_interval == 0)
            if is_full_process:
                if mode == "person":
                    annotated_frame = frame_processor.process_frame(frame, is_full_process=True)
                else:
                    annotated_frame, _ = pipeline.process(frame, frame_no=frame_count)

                processor.write_frame(annotated_frame)

            frame_count += 1

        print(f"Completed: {frame_count} total frames iterated")
        print(f"Processed video saved to: {processor.output_path}")

        if mode == "dual":
            # サマリ出力（現在値ベース）
            stats = pipeline.finalize()
            print("Summary (dual):", {
                # ※ current_suitcases は「三種類の荷物の合計」を意味する
                "persons": stats["realtime"]["current_persons"],
                "bags_total": stats["realtime"]["current_suitcases"],
                "confirmed": stats["confirmed"],
            })

    except IOError as e:
        print(f"I/O Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
    finally:
        processor.release_resources()
        print("Released video resources.")
