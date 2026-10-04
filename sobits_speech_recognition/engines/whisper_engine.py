import os
import torch
import traceback
import time
from .base_engine import BaseEngine

class WhisperEngine(BaseEngine):
    def __init__(self, node):
        super().__init__(node)
        model_root = os.path.expanduser("~/.sobits_speech_recognition/whisper_models")
        os.environ["HF_HOME"] = model_root
        self.node.declare_parameter('backend', 'whisper')
        self.node.declare_parameter('model_name', 'small')
        self.node.declare_parameter('compute_type', 'float16')
        self.node.declare_parameter('device', '')
        self.node.declare_parameter('language', 'en')
        self.node.declare_parameter('task', 'transcribe')
        self.node.declare_parameter('use_prompt', False)
        self.node.declare_parameter('replace_prompt_whisper', [""])
        # faster-whisper のみ: 声の無い区間を Silero VAD で落とす(無音・雑音で「ご視聴ありがとうございました」などを作らない)
        self.node.declare_parameter('vad_filter', False)
        # faster-whisper のみ: 前の区間の文を次の区間のヒントにしない(同じ語のくり返しを防ぐ)
        self.node.declare_parameter('condition_on_previous_text', True)

        self.backend = self.node.get_parameter('backend').value
        self.model_name = self.node.get_parameter('model_name').value
        self.compute_type = self.node.get_parameter('compute_type').value
        self.device_pref = self.node.get_parameter('device').value
        
        self.is_streamable = False
        
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            raw_device = self.device_pref if self.device_pref else ("cuda" if torch.cuda.is_available() else "cpu")
            
            model_root = os.path.expanduser("~/.sobits_speech_recognition/whisper_models")
            os.makedirs(model_root, exist_ok=True)

            if self.backend == "whisper":
                import whisper
                self.model = whisper.load_model(self.model_name, device=raw_device, download_root=model_root)
                self.logger.info(f"[{self.backend}] Model '{self.model_name}' loaded on {raw_device}")
            
            elif self.backend == "faster-whisper":
                from faster_whisper import WhisperModel
                fw_device = "cuda" if "cuda" in raw_device else "cpu"
                load_name = self.model_name
                if "distil" in self.model_name and "/" not in self.model_name:
                    load_name = f"Systran/faster-{self.model_name}"

                self.model = WhisperModel(
                    load_name, 
                    device=fw_device, 
                    compute_type=self.compute_type, 
                    download_root=model_root,
                    local_files_only=False 
                )
                self.logger.info(f"[{self.backend}] Model '{load_name}' loaded on {fw_device}")

        except Exception as e:
            self.logger.fatal(f"Failed to load Whisper model: {e}\n{traceback.format_exc()}")
            raise e

    def transcribe(self, audio_path):
        if self.model is None:
            return "Error: Model not loaded"

        language = self.node.get_parameter('language').value
        task = self.node.get_parameter('task').value
        use_prompt = self.node.get_parameter('use_prompt').value
        prompt_list = self.node.get_parameter('replace_prompt_whisper').value
        prompt_text = " ".join(prompt_list) if use_prompt else ""

        try:
            if self.backend == "whisper":
                result = self.model.transcribe(
                    audio_path,
                    language=language,
                    task=task,
                    initial_prompt=prompt_text if prompt_text else None
                )
                return result.get("text", "").strip()
            
            elif self.backend == "faster-whisper":
                segments, _ = self.model.transcribe(
                    audio_path,
                    language=language,
                    task=task,
                    initial_prompt=prompt_text if prompt_text else None,
                    vad_filter=self.node.get_parameter('vad_filter').value,
                    condition_on_previous_text=self.node.get_parameter('condition_on_previous_text').value,
                )
                return " ".join([s.text for s in segments]).strip()

        except Exception as e:
            self.logger.error(f"Transcription error: {e}\n{traceback.format_exc()}")
            return ""