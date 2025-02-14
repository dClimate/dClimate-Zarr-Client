import io
from Crypto.Cipher import ChaCha20_Poly1305
from Crypto.Random import get_random_bytes
from numcodecs.abc import Codec


class EncryptionCodec(Codec):
    codec_id = "xchacha20poly1305"
    _encryption_key = None

    def __init__(self, header: str):
        self.header = header
        self._encoded_header = header.encode()
        if self._encryption_key is None:
            raise ValueError("Encryption key must be set before using EncryptionCodec.")

    @classmethod
    def set_encryption_key(cls, encryption_key: str):
        """Set the encryption key dynamically (once per runtime)."""
        if not isinstance(encryption_key, str):
            raise ValueError("Encryption key must be a string")
        if not all(c in "0123456789abcdefABCDEF" for c in encryption_key):
            raise ValueError("Encryption key must be a hexadecimal string")
        if len(encryption_key) != 64:  # 32 bytes = 64 hex chars
            raise ValueError("Encryption key must be 32 bytes (64 hex characters)")
        cls._encryption_key = bytes.fromhex(encryption_key)

    def encode(self, buf):
        raw = io.BytesIO()
        raw.write(buf)
        nonce = get_random_bytes(24)  # XChaCha uses 24-byte (192-bit) nonce
        cipher = ChaCha20_Poly1305.new(key=self._encryption_key, nonce=nonce)
        cipher.update(self._encoded_header)
        ciphertext, tag = cipher.encrypt_and_digest(raw.getbuffer())

        return nonce + tag + ciphertext

    def decode(self, buf, out=None):
        nonce, tag, ciphertext = buf[:24], buf[24:40], buf[40:]
        cipher = ChaCha20_Poly1305.new(key=self._encryption_key, nonce=nonce)
        cipher.update(self._encoded_header)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext

    @classmethod
    def from_config(cls, config):
        """Prevent requiring encryption key from metadata."""
        header = config.get("header", "dClimate-Zarr")
        return cls(header=header)
