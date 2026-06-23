"""
Native RTSP capture for cameras that require auth FFmpeg can't do.

OpenCV's bundled FFmpeg (and even FFmpeg 7.1) only implement **MD5** digest for
RTSP. Many modern IP cameras (e.g. Hikvision/Prama firmware) now demand
**SHA-256** digest (RFC 7616) and reject everything else with 401 — so neither
``cv2.VideoCapture`` nor an ``ffmpeg`` subprocess can open the stream.

This module does the RTSP control channel itself (OPTIONS / DESCRIBE / SETUP /
PLAY) with full digest auth — MD5 **or** SHA-256, with or without ``qop`` — then
receives RTP over the interleaved TCP channel, reassembles H.264 Annex-B NAL
units (single / STAP-A / FU-A), and pipes them to FFmpeg purely as a *decoder*
(no networking) to produce BGR frames.

It exposes the small slice of the ``cv2.VideoCapture`` API the camera service
uses — ``isOpened()`` / ``read()`` / ``release()`` — so it drops straight in as
a capture backend. H.264 only (the common CCTV main/sub stream codec).
"""

import base64
import hashlib
import logging
import re
import socket
import subprocess
import threading
import time
from typing import Optional, Tuple
from urllib.parse import unquote, urlsplit

import numpy as np

logger = logging.getLogger(__name__)

_START_CODE = b"\x00\x00\x00\x01"


def _ffmpeg_exe() -> str:
    """Path to a modern FFmpeg used only for H.264 decoding (no networking)."""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class _Digest:
    """RFC 7616 digest responder supporting MD5 and SHA-256, ±qop."""

    def __init__(self, params: dict, user: str, pw: str):
        self.user, self.pw = user, pw
        self.realm = params.get("realm", "")
        self.nonce = params.get("nonce", "")
        self.qop = params.get("qop", "")
        self.opaque = params.get("opaque")
        algo = (params.get("algorithm") or "MD5").upper()
        self._h = hashlib.sha256 if "SHA-256" in algo else hashlib.md5
        self._algo = "SHA-256" if "SHA-256" in algo else "MD5"
        self._nc = 0

    def _hash(self, s: str) -> str:
        return self._h(s.encode()).hexdigest()

    def header(self, method: str, uri: str) -> str:
        ha1 = self._hash(f"{self.user}:{self.realm}:{self.pw}")
        ha2 = self._hash(f"{method}:{uri}")
        parts = [
            f'username="{self.user}"', f'realm="{self.realm}"',
            f'nonce="{self.nonce}"', f'uri="{uri}"', f'algorithm={self._algo}',
        ]
        if self.qop:
            self._nc += 1
            nc = f"{self._nc:08x}"
            cnonce = hashlib.md5(f"{time.time()}{uri}".encode()).hexdigest()[:16]
            resp = self._hash(f"{ha1}:{self.nonce}:{nc}:{cnonce}:auth:{ha2}")
            parts += [f'response="{resp}"', "qop=auth", f"nc={nc}", f'cnonce="{cnonce}"']
        else:
            resp = self._hash(f"{ha1}:{self.nonce}:{ha2}")
            parts.append(f'response="{resp}"')
        if self.opaque is not None:
            parts.append(f'opaque="{self.opaque}"')
        return "Digest " + ", ".join(parts)


class RtspNativeCapture:
    """``cv2.VideoCapture``-compatible RTSP reader for SHA-256/MD5-digest cameras."""

    def __init__(self, url: str, transport: str = "tcp", open_timeout: float = 15.0):
        self._url = url
        self._open_timeout = open_timeout
        self._sock: Optional[socket.socket] = None
        self._dec: Optional[subprocess.Popen] = None
        self._opened = False
        self._running = False
        self._threads: list[threading.Thread] = []

        self._width = 0
        self._height = 0
        self._frame_bytes = 0
        self._latest: Optional[np.ndarray] = None
        self._frame_id = 0
        self._last_read_id = 0
        self._cond = threading.Condition()

        try:
            self._open()
        except Exception as exc:  # noqa: BLE001 — surface as "not opened"
            logger.warning("Native RTSP open failed: %s", exc)
            self.release()

    # ── cv2.VideoCapture-compatible surface ──────────────────

    def isOpened(self) -> bool:  # noqa: N802 — match cv2 API
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Block until a frame newer than the last one read, or stream ends."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: not self._running or self._frame_id > self._last_read_id,
                timeout=10.0,
            )
            if not ok or not self._running or self._latest is None:
                return False, None
            self._last_read_id = self._frame_id
            return True, self._latest

    def release(self) -> None:  # noqa: D401
        self._running = False
        self._opened = False
        with self._cond:
            self._cond.notify_all()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._dec is not None:
            for stream in (self._dec.stdin, self._dec.stdout, self._dec.stderr):
                try:
                    stream and stream.close()
                except OSError:
                    pass
            try:
                self._dec.kill()
            except OSError:
                pass
            self._dec = None

    # ── RTSP setup ───────────────────────────────────────────

    def _open(self) -> None:
        u = urlsplit(self._url)
        host = u.hostname or ""
        port = u.port or 554
        user = unquote(u.username or "")
        pw = unquote(u.password or "")
        # URL without credentials — what we put on the RTSP request line.
        base = f"rtsp://{host}:{port}{u.path}"
        if u.query:
            base += f"?{u.query}"

        self._sock = socket.create_connection((host, port), timeout=self._open_timeout)
        self._sock.settimeout(self._open_timeout)
        self._cseq = 0
        digest: Optional[_Digest] = None

        def request(method: str, uri: str, extra: str = "") -> Tuple[str, str]:
            self._cseq += 1
            req = f"{method} {uri} RTSP/1.0\r\nCSeq: {self._cseq}\r\n"
            if digest is not None:
                req += f"Authorization: {digest.header(method, uri)}\r\n"
            req += extra + "\r\n"
            self._sock.sendall(req.encode())
            return self._recv_reply()

        request("OPTIONS", base)
        head, _ = request("DESCRIBE", base, "Accept: application/sdp\r\n")
        if "401" in head.split("\r\n", 1)[0]:
            m = re.search(r"WWW-Authenticate:\s*Digest\s+(.*)", head, re.I)
            if not m:
                raise RuntimeError("camera requires non-digest auth")
            params = dict(re.findall(r'(\w+)="?([^",\r\n]*)"?', m.group(1)))
            digest = _Digest(params, user, pw)
            head, sdp = request("DESCRIBE", base, "Accept: application/sdp\r\n")
        else:
            sdp = _  # body from first DESCRIBE
        if "200" not in head.split("\r\n", 1)[0]:
            raise RuntimeError(f"DESCRIBE failed: {head.splitlines()[0]}")

        if "H264" not in sdp and "h264" not in sdp:
            raise RuntimeError("native RTSP path supports H.264 only")

        sps, pps = self._parse_parameter_sets(sdp)
        cbase = (re.search(r"Content-Base:\s*(\S+)", head, re.I) or [None, base + "/"])[1]
        track_url = self._track_url(sdp, cbase, base)

        head, _ = request("SETUP", track_url,
                          "Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n")
        if "200" not in head.split("\r\n", 1)[0]:
            raise RuntimeError(f"SETUP failed: {head.splitlines()[0]}")
        session = (re.search(r"Session:\s*([^\r\n;]+)", head, re.I) or [None, ""])[1].strip()

        head, _ = request("PLAY", base,
                          f"Session: {session}\r\nRange: npt=0.000-\r\n")
        if "200" not in head.split("\r\n", 1)[0]:
            raise RuntimeError(f"PLAY failed: {head.splitlines()[0]}")

        self._start_decoder()
        self._running = True
        self._spawn(self._stderr_loop, daemon=True)
        self._spawn(self._stdout_loop, daemon=True)
        self._spawn(self._rtp_loop, args=(sps, pps), daemon=True)

        # Wait until the decoder reports a resolution and we have a frame.
        deadline = time.time() + self._open_timeout
        with self._cond:
            self._cond.wait_for(
                lambda: not self._running or self._frame_id > 0,
                timeout=max(0.1, deadline - time.time()),
            )
        self._opened = self._running and self._frame_id > 0
        if not self._opened:
            raise RuntimeError("no frame decoded within timeout")

    def _recv_reply(self) -> Tuple[str, str]:
        data = b""
        while b"\r\n\r\n" not in data:
            c = self._sock.recv(4096)
            if not c:
                break
            data += c
        head, _, rest = data.partition(b"\r\n\r\n")
        text = head.decode("utf-8", "ignore")
        m = re.search(r"Content-Length:\s*(\d+)", text, re.I)
        body = rest
        if m:
            need = int(m.group(1))
            while len(body) < need:
                c = self._sock.recv(4096)
                if not c:
                    break
                body += c
        return text, body.decode("utf-8", "ignore")

    @staticmethod
    def _parse_parameter_sets(sdp: str) -> Tuple[bytes, bytes]:
        sp = re.search(r"sprop-parameter-sets=([^;\r\n ]+)", sdp)
        if not sp:
            return b"", b""
        parts = sp.group(1).split(",")
        try:
            sps = base64.b64decode(parts[0]) if parts[0] else b""
            pps = base64.b64decode(parts[1]) if len(parts) > 1 and parts[1] else b""
        except Exception:  # noqa: BLE001
            return b"", b""
        return sps, pps

    @staticmethod
    def _track_url(sdp: str, cbase: str, base: str) -> str:
        ctrl = "trackID=1"
        for line in sdp.splitlines():
            if line.startswith("a=control:") and "*" not in line:
                ctrl = line.split(":", 1)[1].strip()
        if ctrl.startswith("rtsp://"):
            return ctrl
        return cbase.rstrip("/") + "/" + ctrl

    # ── Decode pipeline ──────────────────────────────────────

    def _start_decoder(self) -> None:
        self._dec = subprocess.Popen(
            [_ffmpeg_exe(), "-nostdin", "-loglevel", "info", "-fflags", "nobuffer",
             "-f", "h264", "-i", "pipe:0",
             "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def _spawn(self, target, args=(), daemon=True) -> None:
        t = threading.Thread(target=target, args=args, daemon=daemon)
        t.start()
        self._threads.append(t)

    def _stderr_loop(self) -> None:
        """Drain FFmpeg logs; capture the decoded resolution."""
        if not self._dec or not self._dec.stderr:
            return
        for raw in self._dec.stderr:
            if not self._running:
                break
            line = raw.decode("utf-8", "ignore")
            if self._frame_bytes == 0:
                m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", line)
                if m:
                    self._width, self._height = int(m.group(1)), int(m.group(2))
                    self._frame_bytes = self._width * self._height * 3

    def _stdout_loop(self) -> None:
        """Read decoded BGR frames and publish the newest."""
        while self._running and self._frame_bytes == 0:
            time.sleep(0.02)
        if not self._dec or not self._dec.stdout or self._frame_bytes == 0:
            return
        stdout = self._dec.stdout
        need = self._frame_bytes
        while self._running:
            buf = stdout.read(need)
            if not buf or len(buf) < need:
                break
            # Copy out of the read-only pipe buffer so downstream in-place CV ops
            # (resize, annotation) have a writable, contiguous frame.
            frame = np.frombuffer(buf, np.uint8).reshape(
                (self._height, self._width, 3)).copy()
            with self._cond:
                self._latest = frame
                self._frame_id += 1
                self._cond.notify_all()
        self._running = False
        with self._cond:
            self._cond.notify_all()

    def _rtp_loop(self, sps: bytes, pps: bytes) -> None:
        """Receive interleaved RTP, reassemble NAL units, feed the decoder."""
        sent_params = False
        fu_buf = bytearray()

        def recv_exact(n: int) -> Optional[bytes]:
            buf = b""
            while len(buf) < n:
                try:
                    c = self._sock.recv(n - len(buf))
                except OSError:
                    return None
                if not c:
                    return None
                buf += c
            return buf

        def feed(nal: bytes) -> None:
            try:
                if self._dec and self._dec.stdin:
                    self._dec.stdin.write(_START_CODE + nal)
            except (BrokenPipeError, OSError):
                pass

        while self._running:
            b0 = recv_exact(1)
            if not b0:
                break
            if b0 != b"$":
                continue  # RTSP message or resync byte
            hdr = recv_exact(3)
            if not hdr:
                break
            channel = hdr[0]
            length = (hdr[1] << 8) | hdr[2]
            pkt = recv_exact(length)
            if not pkt or channel != 0 or length < 13:
                continue
            cc = pkt[0] & 0x0F
            payload = pkt[12 + 4 * cc:]
            if not payload:
                continue
            if not sent_params and sps and pps:
                feed(sps); feed(pps); sent_params = True
            ntype = payload[0] & 0x1F
            if ntype <= 23:
                feed(payload)
            elif ntype == 24:  # STAP-A
                i = 1
                while i + 2 <= len(payload):
                    sz = (payload[i] << 8) | payload[i + 1]
                    i += 2
                    feed(payload[i:i + sz]); i += sz
            elif ntype == 28:  # FU-A
                fu = payload[1]
                if fu & 0x80:  # start
                    fu_buf = bytearray([(payload[0] & 0xE0) | (fu & 0x1F)])
                fu_buf.extend(payload[2:])
                if fu & 0x40:  # end
                    feed(bytes(fu_buf)); fu_buf = bytearray()
        self._running = False
        with self._cond:
            self._cond.notify_all()
