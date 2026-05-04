"""XOR decryption utilities for BrowseComp dataset.

Ported from the official OpenAI simple-evals implementation.
"""

import base64
import hashlib


def derive_key(password: str, length: int) -> bytes:
    """Derive an XOR key by repeating the SHA-256 hash of *password*.

    Args:
        password: The passphrase to derive the key from.
        length: Desired key length in bytes.

    Returns:
        Bytes of exactly *length* bytes.
    """
    digest = hashlib.sha256(password.encode()).digest()
    return (digest * (length // len(digest) + 1))[:length]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Base64-decode *ciphertext_b64* then XOR-decrypt with *password*.

    Args:
        ciphertext_b64: Base64-encoded ciphertext string.
        password: The passphrase used to derive the XOR key.

    Returns:
        The decrypted UTF-8 string.
    """
    data = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(data))
    return bytes(a ^ b for a, b in zip(data, key)).decode("utf-8")


def encrypt(plaintext: str, password: str) -> str:
    """XOR-encrypt *plaintext* with *password* and return a Base64-encoded string.

    Args:
        plaintext: The UTF-8 string to encrypt.
        password: The passphrase used to derive the XOR key.

    Returns:
        Base64-encoded ciphertext string (symmetric with ``decrypt``).
    """
    data = plaintext.encode("utf-8")
    key = derive_key(password, len(data))
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key))).decode("ascii")
