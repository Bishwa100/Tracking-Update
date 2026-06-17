"""
Singleton ML model manager.
Loads YOLOv8n (person detection), InsightFace/ArcFace (512-d face embeddings),
and OSNet x0.25 (512-d body re-ID) once on startup. Runs on CPU or CUDA GPU;
the device is chosen at load time and can be switched live via reload().
"""

import logging
import os
from typing import Optional, List

# Disable Ultralytics runtime pip auto-install (see app/main.py for the full
# rationale) — must be set before `ultralytics` is first imported below.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import cv2
import numpy as np
import torch

from app.config import settings

logger = logging.getLogger(__name__)


class FaceEmbeddingCache:
    """
    Per-stream ArcFace embedding cache keyed by a dHash of the ALIGNED face
    crop. A face that is pixel-stable across frames is embedded once and reused
    at zero inference cost. Exact-hash matching means a collision requires two
    visually identical aligned crops — which would embed identically anyway.
    """

    def __init__(self):
        self._store: dict[int, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def get(self, signature: int) -> Optional[np.ndarray]:
        emb = self._store.get(signature)
        if emb is not None:
            self.hits += 1
        return emb

    def put(self, signature: int, embedding: np.ndarray) -> None:
        self.misses += 1
        self._store[signature] = embedding


def filter_persons(detections: List[dict], min_conf: float = 0.4) -> List[dict]:
    """Filter a detect_all() result down to person-class boxes (COCO class 0)."""
    return [
        d for d in detections
        if d.get("class_id") == 0 and d.get("confidence", 0.0) >= min_conf
    ]


def cuda_available() -> bool:
    """True when a usable CUDA GPU is visible to torch (CUDA torch build present)."""
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def gpu_info() -> dict:
    """Name + total/used VRAM (MB) for GPU 0, or {} when no CUDA device."""
    if not cuda_available():
        return {}
    try:
        props = torch.cuda.get_device_properties(0)
        total_mb = int(props.total_memory / (1024 * 1024))
        used_mb = int(torch.cuda.memory_allocated(0) / (1024 * 1024))
        return {"name": props.name, "memory_mb": total_mb, "memory_used_mb": used_mb}
    except Exception:
        return {}


def resolve_device(requested: str) -> str:
    """
    Map a requested device ("auto" | "cpu" | "cuda"/"gpu") to a concrete device
    string, honouring actual hardware availability.

    - "auto": "cuda" if a CUDA GPU is available, else "cpu".
    - "cuda"/"gpu": "cuda" if available, else "cpu" (logs a warning).
    - anything else: "cpu".
    """
    req = (requested or "auto").strip().lower()
    if req == "auto":
        return "cuda" if cuda_available() else "cpu"
    if req in ("cuda", "gpu"):
        if cuda_available():
            return "cuda"
        logger.warning("Device 'cuda' requested but no CUDA GPU is available — using CPU.")
        return "cpu"
    return "cpu"


class ModelManager:
    """
    Singleton holding all pre-trained models in CPU memory.
    Call `load_all()` once during FastAPI lifespan startup.
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.yolo = None
        self.face_app = None
        self.body_model_type = "none"  # "osnet" | "none"
        self.osnet_model = None
        self.device = "cpu"
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Loading ──────────────────────────────────────────────

    def load_all(
        self,
        yolo_path: str = "yolov8n.pt",
        insightface_name: str = "buffalo_l",
        device: str = "auto",
    ):
        """Load all models into memory on `device`. Call once on startup."""
        if self._loaded:
            logger.info("Models already loaded, skipping.")
            return

        self.device = resolve_device(device)
        logger.info("Loading models on device: %s", self.device)
        self._load_yolo(yolo_path)
        self._load_arcface(insightface_name)
        self._load_body_model()
        self._warmup()

        self._loaded = True
        logger.info("All models loaded and warmed up successfully.")

    def reload(
        self,
        device: str,
        yolo_path: Optional[str] = None,
        insightface_name: Optional[str] = None,
    ) -> dict:
        """
        Tear down all models and reload them on a new device. Blocking/CPU-GPU
        heavy — callers must run this off the event loop and pause inference
        (hold the inference semaphore) while it runs. Returns the new status().
        """
        prev_device = self.device
        logger.info("Reloading models: %s -> %s", prev_device, device)

        # Drop references so the old (possibly GPU-resident) graphs are freed.
        self.yolo = None
        self.face_app = None
        self.osnet_model = None
        self.body_model_type = "none"
        self._loaded = False
        if prev_device == "cuda" and cuda_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        self.load_all(
            yolo_path=yolo_path or settings.YOLO_MODEL_PATH,
            insightface_name=insightface_name or settings.INSIGHTFACE_MODEL_NAME,
            device=device,
        )
        return self.status()

    def _load_yolo(self, model_path: str):
        from ultralytics import YOLO

        # On GPU, run the native PyTorch model (.pt) — it executes on CUDA via the
        # device= arg at predict time. The ONNX fast-path below is CPU-only.
        if self.device == "cuda":
            logger.info("Loading YOLOv8n (PyTorch, CUDA) from %s...", model_path)
            self.yolo = YOLO(model_path)
            logger.info("YOLOv8n loaded (CUDA).")
            return

        # Prefer an ONNX graph on CPU (typically 2-3x faster). Export once and
        # cache next to the .pt file; fall back to .pt if export/load fails.
        if settings.YOLO_USE_ONNX and model_path.endswith(".pt"):
            from pathlib import Path

            onnx_path = str(Path(model_path).with_suffix(".onnx"))
            try:
                if not Path(onnx_path).exists():
                    logger.info("Exporting YOLO to ONNX (%s)...", onnx_path)
                    YOLO(model_path).export(format="onnx", imgsz=640)
                logger.info("Loading YOLO (ONNX) from %s...", onnx_path)
                self.yolo = YOLO(onnx_path, task="detect")
                logger.info("YOLO loaded (onnxruntime CPU).")
                return
            except Exception as exc:
                logger.warning(
                    "YOLO ONNX export/load failed (%s) — falling back to PyTorch.",
                    exc,
                )

        logger.info("Loading YOLOv8n from %s...", model_path)
        self.yolo = YOLO(model_path)
        logger.info("YOLOv8n loaded.")

    def _load_arcface(self, model_name: str):
        from insightface.app import FaceAnalysis

        use_cuda = self.device == "cuda"
        # Keep CPU as a fallback provider so a missing/mismatched onnxruntime-gpu
        # (cuDNN, etc.) degrades to CPU for faces rather than crashing the load.
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda
            else ["CPUExecutionProvider"]
        )
        logger.info("Loading InsightFace (%s) on %s...", model_name, self.device)
        self.face_app = FaceAnalysis(
            name=model_name,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        det = settings.INSIGHTFACE_DET_SIZE
        self.face_app.prepare(ctx_id=0 if use_cuda else -1, det_size=(det, det))
        logger.info("InsightFace/ArcFace loaded (%s, det_size=%d).", self.device, det)

    def _load_body_model(self):
        body_type = settings.BODY_MODEL_TYPE.strip().lower()
        if body_type != "osnet":
            logger.info("Body embedding stream disabled (BODY_MODEL_TYPE=%s).", body_type)
            return
        try:
            self._load_osnet()
        except Exception as exc:
            self.osnet_model = None
            logger.warning(
                "OSNet could not be loaded (%s) — body stream disabled, "
                "running face-only.",
                exc,
            )

    def _load_osnet(self):
        from pathlib import Path
        from urllib.request import Request, urlopen

        from app.osnet import osnet_x0_25, load_pretrained_weights

        weights_path = Path(settings.OSNET_WEIGHTS_PATH)
        if not weights_path.exists():
            url = settings.OSNET_WEIGHTS_URL
            if not url:
                raise FileNotFoundError(
                    f"OSNet weights not found at {weights_path} and no "
                    "OSNET_WEIGHTS_URL configured."
                )
            logger.info("Downloading OSNet weights from %s ...", url)
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            request = Request(url, headers={"User-Agent": "RestaurantTracker/1.0"})
            with urlopen(request, timeout=60) as response:
                data = response.read()
            tmp_path = weights_path.with_suffix(".tmp")
            tmp_path.write_bytes(data)
            tmp_path.replace(weights_path)
            logger.info("OSNet weights saved to %s (%d bytes).", weights_path, len(data))

        model = osnet_x0_25(num_classes=1, loss="softmax")
        matched = load_pretrained_weights(model, str(weights_path))
        if matched == 0:
            raise RuntimeError(f"Checkpoint {weights_path} is incompatible with OSNet x0.25.")
        self.osnet_model = model.to(self.device).eval()
        self.body_model_type = "osnet"
        logger.info("OSNet x0.25 body model loaded (512-d re-ID embeddings, %s).", self.device)

    def _warmup(self):
        """Run dummy inference through all models to warm up JIT/graph."""
        logger.info("Warming up models...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)

        if self.yolo:
            self.yolo.predict(dummy, verbose=False, conf=0.5, classes=[0], device=self.device)
        if self.face_app:
            self.face_app.get(dummy)
        if self.osnet_model is not None:
            self.extract_body_embeddings([dummy[:256, :128]])

        logger.info("Warm-up complete.")

    # ── Inference ────────────────────────────────────────────

    def detect_persons(self, image: np.ndarray, confidence: float = 0.5) -> List[dict]:
        """Run YOLOv8n person detection. Returns [{bbox, confidence}]."""
        results = self.yolo.predict(
            image, verbose=False, conf=confidence, classes=[0], device=self.device,
        )
        persons = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                persons.append({
                    "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                    "confidence": conf,
                })
        return persons

    def _upscale_for_arcface(self, image: np.ndarray, min_dim: int = 112) -> np.ndarray:
        """Upscale a tiny crop (ArcFace struggles below ~112px)."""
        h, w = image.shape[:2]
        if h < min_dim or w < min_dim:
            scale = max(min_dim / h, min_dim / w)
            return cv2.resize(
                image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
            )
        return image

    def extract_face_data(
        self,
        image: np.ndarray,
        embedding_cache: Optional[FaceEmbeddingCache] = None,
    ) -> Optional[dict]:
        """Detect the highest-scoring face in an image. Returns dict or None."""
        image = self._upscale_for_arcface(image)
        faces = self.extract_all_faces(image, embedding_cache=embedding_cache)
        if not faces:
            return None
        return max(faces, key=lambda f: f["det_score"])

    def extract_all_faces(
        self,
        image: np.ndarray,
        embedding_cache: Optional[FaceEmbeddingCache] = None,
    ) -> List[dict]:
        """
        Detect EVERY face in an image in a single ArcFace pass.
        Returns [{embedding, bbox, det_score}] in frame coordinates.
        """
        if embedding_cache is not None:
            try:
                return self._extract_all_faces_cached(image, embedding_cache)
            except Exception as exc:
                logger.warning(
                    "Cached face extraction failed (%s) — using standard pass.", exc
                )

        faces = self.face_app.get(image)
        results: List[dict] = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            results.append({
                "embedding": face.normed_embedding,
                "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "det_score": float(face.det_score),
            })
        return results

    def _extract_all_faces_cached(
        self, image: np.ndarray, embedding_cache: FaceEmbeddingCache
    ) -> List[dict]:
        """Detection + per-face cached recognition (skips ArcFace on a cache hit)."""
        from insightface.utils import face_align

        from app.utils import compute_dhash

        det_model = getattr(self.face_app, "det_model", None)
        rec_model = (self.face_app.models or {}).get("recognition")
        if det_model is None or rec_model is None:
            raise RuntimeError("detection/recognition models unavailable")

        bboxes, kpss = det_model.detect(image, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return []
        if kpss is None:
            raise RuntimeError("detector returned no landmarks")

        input_size = getattr(rec_model, "input_size", None) or (112, 112)
        results: List[dict] = []
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i, 0:4].astype(int)
            det_score = float(bboxes[i, 4])
            aligned = face_align.norm_crop(
                image, landmark=kpss[i], image_size=input_size[0]
            )
            sig = compute_dhash(aligned, hash_size=8)
            embedding = embedding_cache.get(sig)
            if embedding is None:
                embedding = np.asarray(rec_model.get_feat(aligned)).flatten()
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                embedding_cache.put(sig, embedding)
            results.append({
                "embedding": embedding,
                "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "det_score": det_score,
            })
        return results

    # ImageNet normalization used by torchreid at train and test time.
    _OSNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _OSNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def extract_body_embedding(self, person_crop: np.ndarray) -> np.ndarray:
        return self.extract_body_embeddings([person_crop])[0]

    def extract_body_embeddings(self, person_crops: List[np.ndarray]) -> List[np.ndarray]:
        """Batched OSNet body-embedding extraction (one forward pass for all crops)."""
        if not person_crops:
            return []
        if self.osnet_model is None:
            raise RuntimeError("No body embedding model is loaded.")

        batch = np.stack([
            (
                cv2.cvtColor(
                    cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR),
                    cv2.COLOR_BGR2RGB,
                ).astype(np.float32) / 255.0
                - self._OSNET_MEAN
            ) / self._OSNET_STD
            for crop in person_crops
        ])  # (N, 256, 128, 3)
        tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(self.device)

        with torch.no_grad():
            features = self.osnet_model(tensor)  # (N, 512)

        embeddings = features.cpu().numpy()
        results: List[np.ndarray] = []
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            results.append(emb)
        return results

    # ── Status ───────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def has_body_model(self) -> bool:
        return self.osnet_model is not None

    def status(self) -> dict:
        return {
            "yolo_loaded": self.yolo is not None,
            "arcface_loaded": self.face_app is not None,
            "body_model": self.body_model_type if self.has_body_model else "none",
            "device": self.device,
        }
