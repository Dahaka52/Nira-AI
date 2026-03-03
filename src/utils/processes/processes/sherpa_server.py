import logging
import asyncio
import os
import subprocess
from typing import Set
from utils.config import Config
from utils.processes.base import BaseProcess

class SherpaSTTProcess(BaseProcess):
    def __init__(self):
        super().__init__("sherpa_stt")
        self._log_file = None

    def _close_log_file(self):
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    async def unload(self):
        try:
            await super().unload()
        finally:
            self._close_log_file()

    def _read_token_set(self, tokens_path: str) -> Set[str]:
        token_set: Set[str] = set()
        with open(tokens_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                # tokens.txt format: "<token> <id>"
                token = line.split(maxsplit=1)[0]
                token_set.add(token)
        return token_set

    def _prepare_hotwords_file(self, hotwords_file: str, tokens_path: str, output_path: str) -> str:
        """
        Validate hotwords tokens against current model vocabulary.
        Returns path to sanitized file, or empty string if nothing valid.
        """
        if not hotwords_file or not os.path.exists(hotwords_file):
            return ""

        token_set = self._read_token_set(tokens_path)
        valid_lines = []
        invalid_count = 0

        # utf-8-sig removes optional BOM if file was edited by Windows tools
        with open(hotwords_file, "r", encoding="utf-8-sig") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                pieces = line.split()
                if not pieces:
                    continue
                if all(piece in token_set for piece in pieces):
                    valid_lines.append(" ".join(pieces))
                else:
                    invalid_count += 1
                    logging.warning("Sherpa STT hotword line skipped (unknown token): %s", line)

        if not valid_lines:
            logging.warning(
                "Sherpa STT hotwords disabled: no valid entries in file %s (invalid lines: %s)",
                hotwords_file,
                invalid_count,
            )
            return ""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(valid_lines) + "\n")

        if invalid_count > 0:
            logging.warning(
                "Sherpa STT hotwords sanitized: %s valid, %s invalid",
                len(valid_lines),
                invalid_count,
            )
        else:
            logging.info("Sherpa STT hotwords validated: %s entries", len(valid_lines))

        return output_path
        
    async def reload(self):
        await self.unload()
        
        config = Config()
        
        # We rely on the install.bat script placing this right
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        server_script = os.path.abspath(os.path.join(script_dir, "apps", "stt-sherpa-server", "start_server.py"))
        model_dir = os.path.abspath(os.path.join(script_dir, "models", "vosk-model-small-streaming-ru"))
        
        if not os.path.exists(server_script):
            logging.error(f"Sherpa STT server script not found at {server_script}")
            raise FileNotFoundError(f"Sherpa server script missing: {server_script}")

        if not os.path.exists(model_dir):
            logging.error(f"Sherpa STT model not found at {model_dir}")
            raise FileNotFoundError(f"Sherpa model missing: {model_dir}")
            
        provider = "cuda"
        gpu_id = 0
        model_variant = "int8"
        decoding_method = "modified_beam_search"
        num_active_paths = 4
        use_endpoint = 0
        hotwords_file = ""
        hotwords_score = 1.5
        bpe_vocab_path = ""
        encoder_path = None
        decoder_path = None
        joiner_path = None
        tokens_path = None
        # Find sherpa config inside operations list
        for op in config.operations:
            if op.get("id") == "sherpa" and op.get("role") == "stt":
                provider = op.get("provider", "cuda")
                try:
                    gpu_id = int(op.get("gpu_id", 0))
                except Exception:
                    gpu_id = 0
                model_dir = os.path.abspath(str(op.get("model_dir", model_dir)))
                model_variant = str(op.get("model_variant", "int8")).lower().strip()
                decoding_method = str(op.get("decoding_method", "modified_beam_search")).strip()
                try:
                    num_active_paths = int(op.get("num_active_paths", 4))
                except Exception:
                    num_active_paths = 4
                try:
                    use_endpoint = int(op.get("use_endpoint", 0))
                except Exception:
                    use_endpoint = 0
                hotwords_file = str(op.get("hotwords_file", "") or "").strip()
                try:
                    hotwords_score = float(op.get("hotwords_score", 1.5))
                except Exception:
                    hotwords_score = 1.5
                bpe_vocab_path = str(op.get("bpe_vocab", "") or "").strip()

                encoder_path = op.get("encoder", None)
                decoder_path = op.get("decoder", None)
                joiner_path = op.get("joiner", None)
                tokens_path = op.get("tokens", None)
                break

        if not os.path.exists(model_dir):
            logging.error(f"Sherpa STT model not found at {model_dir}")
            raise FileNotFoundError(f"Sherpa model missing: {model_dir}")

        if not encoder_path:
            if model_variant in ("fp32", "full", "onnx"):
                encoder_path = os.path.join(model_dir, "am-onnx", "encoder.onnx")
            elif model_variant in ("chunk64", "chunk"):
                encoder_path = os.path.join(model_dir, "am-onnx", "encoder.chunk64.onnx")
            else:
                encoder_path = os.path.join(model_dir, "am-onnx", "encoder.int8.onnx")
        if not decoder_path:
            decoder_path = os.path.join(model_dir, "am-onnx", "decoder.onnx") if model_variant in ("fp32", "full", "onnx") else os.path.join(model_dir, "am-onnx", "decoder.int8.onnx")
        if not joiner_path:
            joiner_path = os.path.join(model_dir, "am-onnx", "joiner.onnx") if model_variant in ("fp32", "full", "onnx") else os.path.join(model_dir, "am-onnx", "joiner.int8.onnx")
        if not tokens_path:
            tokens_path = os.path.join(model_dir, "lang", "tokens.txt")

        encoder_path = os.path.abspath(str(encoder_path))
        decoder_path = os.path.abspath(str(decoder_path))
        joiner_path = os.path.abspath(str(joiner_path))
        tokens_path = os.path.abspath(str(tokens_path))
        if bpe_vocab_path:
            bpe_vocab_path = os.path.abspath(str(bpe_vocab_path))
        else:
            default_bpe_vocab = os.path.join(model_dir, "lang", "unigram_500.vocab")
            if os.path.exists(default_bpe_vocab):
                bpe_vocab_path = os.path.abspath(default_bpe_vocab)

        for required_path in (encoder_path, decoder_path, joiner_path, tokens_path):
            if not os.path.exists(required_path):
                logging.error("Sherpa STT required file not found: %s", required_path)
                raise FileNotFoundError(f"Sherpa required file missing: {required_path}")

        if bpe_vocab_path and not os.path.exists(bpe_vocab_path):
            logging.warning("Sherpa STT bpe_vocab file not found: %s (ignored)", bpe_vocab_path)
            bpe_vocab_path = ""

        default_hotwords_file = os.path.abspath(os.path.join(script_dir, "models", "hotwords.txt"))
        if not hotwords_file and os.path.exists(default_hotwords_file):
            hotwords_file = default_hotwords_file
        elif hotwords_file:
            hotwords_file = os.path.abspath(hotwords_file)

        # Assemble command
        
        import sys
        cmd = [
            sys.executable,
            server_script,
            "--encoder", encoder_path,
            "--decoder", decoder_path,
            "--joiner", joiner_path,
            "--tokens", tokens_path,
            "--modeling-unit", "bpe",
            "--decoding-method", decoding_method,
            "--num-active-paths", str(max(1, num_active_paths)),
            "--use-endpoint", str(use_endpoint),
            "--port", "6006",
            "--provider", provider,
            "--doc-root", os.path.dirname(server_script) # Pass the script's directory as doc-root to avoid crash
        ]
        if bpe_vocab_path:
            cmd += ["--bpe-vocab", bpe_vocab_path]

        if hotwords_file:
            if os.path.exists(hotwords_file):
                from utils.args import args
                sanitized_hotwords = os.path.join(args.log_dir, "sherpa_hotwords.resolved.txt")
                prepared_hotwords = self._prepare_hotwords_file(
                    hotwords_file=hotwords_file,
                    tokens_path=tokens_path,
                    output_path=sanitized_hotwords,
                )
                if prepared_hotwords:
                    cmd += ["--hotwords-file", prepared_hotwords, "--hotwords-score", str(hotwords_score)]
                else:
                    hotwords_file = ""
            else:
                logging.warning("Sherpa STT hotwords file not found: %s (ignored)", hotwords_file)
                hotwords_file = ""
        
        env = os.environ.copy()
        provider_low = str(provider).lower()
        if provider_low.startswith("cuda"):
            # Pin Sherpa process to a specific GPU index from config.
            # Inside child process it will be visible as CUDA device 0.
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logging.info(
                "Sherpa STT GPU pin: provider=%s, requested gpu_id=%s, CUDA_VISIBLE_DEVICES=%s",
                provider,
                gpu_id,
                env["CUDA_VISIBLE_DEVICES"],
            )
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)

        logging.info(
            "Sherpa STT config: model_dir=%s, model_variant=%s, decoding_method=%s, num_active_paths=%s, use_endpoint=%s, bpe_vocab=%s, hotwords_file=%s, hotwords_score=%s",
            model_dir,
            model_variant,
            decoding_method,
            num_active_paths,
            use_endpoint,
            bpe_vocab_path if bpe_vocab_path else "",
            hotwords_file if hotwords_file else "",
            hotwords_score,
        )
        logging.info(f"Starting Sherpa STT Server with command: {' '.join(cmd)}")
        
        # Направляем вывод в лог для отладки
        from utils.args import args
        log_path = os.path.join(args.log_dir, "sherpa_server.log")
        self._close_log_file()
        self._log_file = open(log_path, "w", encoding="utf-8")

        # Start the background process
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=self._log_file,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0
        )
        
        # Allow time to init
        await asyncio.sleep(4)
        
        if self.process.poll() is not None:
             logging.error(f"Failed to start Sherpa STT Server process (Exit code: {self.process.returncode}).")
             self.process = None
             self._close_log_file()

    async def _close(self):
        if self.process:
            logging.info("Terminating Sherpa STT Server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
