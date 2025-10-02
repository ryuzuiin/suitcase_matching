# libs/preprocessing.py
import cv2
import numpy as np

class VideoProcessor:
    """
    動画ファイルの読み込みと書き込みを管理するクラス。

    属性:
        input_path (str): 入力動画ファイルのパス
        output_path (str): 出力動画ファイルのパス
        target_fps (int): 処理後の目標FPS
        cap (cv2.VideoCapture): 入力動画キャプチャオブジェクト
        out (cv2.VideoWriter): 出力動画書き出しオブジェクト
    """

    def __init__(self, input_path: str, output_path: str, target_fps: int = 10):
        """
        VideoProcessorを初期化する。

        引数:
            input_path (str): 入力動画ファイルのパス
            output_path (str): 出力動画ファイルのパス
            target_fps (int): 処理後の目標FPS
        """
        self.input_path = input_path
        self.output_path = output_path
        self.target_fps = target_fps
        self.cap= None
        self.out = None

    def open_video(self):
        """
        動画ファイルを開き、そのメタデータを取得する。

        このメソッドは動画を開くアクションを実行し、
        オブジェクトの内部状態を変更する。
        値を返して外部で処理させるものではない。
        
        例外:
            IOError: 動画ファイルが開けない場合
        """
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {self.input_path}")
        # 動画のメタデータを取得
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 処理するフレームの間隔を計算 (例: 元が30FPSで目標が10FPSなら、3フレームごとに1フレーム処理)
        self.frame_interval = max(1, self.fps // self.target_fps)

    def setup_output(self):
        '''
        出力動画ファイルを設定し、VideoWriterオブジェクトを作成する。

        例外:
            IOError: 出力動画ファイルが作成できない場合
        '''
        # MP4Vコーデックを試す
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.target_fps, (self.width, self.height))
        if not self.out.isOpened():
            # MP4Vで失敗した場合、XVIDコーデックとAVI形式で再試行
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            alt_output_path = self.output_path.replace('.mp4', '.avi')
            self.out = cv2.VideoWriter(alt_output_path, fourcc, self.target_fps, (self.width, self.height))
            if not self.out.isOpened():
                raise IOError(f"Cannot open video writer for file: {self.output_path} or {alt_output_path}")
            self.output_path = alt_output_path
  
    def read_frame(self) -> tuple:
        '''
        動画ストリームから1フレーム読み込む。

        戻り値:
            tuple: (ret, frame)のタプル。retは成功時にTrue、frameは画像(NumPy配列)。
        
        例外:
            ValueError: 動画キャプチャが初期化されていない場合
        '''
        if not self.cap:
            raise ValueError("VideoCapture is not initialized. Call open_video() first.")
        # 動画からフレームを読み込む
        ret, frame = self.cap.read()
        return ret, frame
    
    def write_frame(self, frame: np.ndarray):
        '''
        処理したフレームを出力動画ファイルに書き込む。

        引数:
            frame (np.ndarray): 書き込むフレーム画像

        例外:
            ValueError: VideoWriterが初期化されていない場合
        '''
        if not self.out or not self.out.isOpened():
            raise IOError("VideoWriter is not initialized. Call setup_output() first.")
        # フレームを出力ファイルに書き込む
        self.out.write(frame)

    def start_processing(self):
        """
        動画処理の開始。入力動画を開き、出力動画を設定する。

        このメソッドは内部で open_video() と setup_output() を呼び出し、
        外部からの一貫した処理開始インターフェースを提供する。
        """
        self.open_video()
        self.setup_output()
        print(f"Video opened. FPS: {self.fps}, Width: {self.width}, Height: {self.height}, Total Frames: {self.total_frames}")
        
    def release_resources(self):
        '''
        動画キャプチャと動画書き出しのオブジェクトを解放する。
        '''
        # 入力動画オブジェクトを解放
        if self.cap:
            self.cap.release()
        # 出力動画オブジェクトを解放
        if hasattr(self, 'out') and self.out:
            self.out.release()
        


        