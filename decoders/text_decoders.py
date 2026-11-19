import re
import base64
import binascii

MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '-----': '0', '.----': '1', '..---': '2', '...--': '3',
    '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8',
    '----.': '9', '.-.-.-': '.', '--..--': ',', '..--..': '?', '-.-.--': '!',
    '-....-': '-', '-..-.': '/', '.--.-.': '@', '-...-': '=', '---...': ':'
}

# Flag pattern supporting custom CTF prefixes (e.g. picoCTF{...}, riftCTF{...}, FLAG{...})
FLAG_REGEX = re.compile(
    r'\b[a-zA-Z0-9_\-]{0,15}(?:FLAG|CTF|KEY|SECRET|PASS|CYBER)\{[A-Za-z0-9_\-!@#\$%\^&\*\+=.,]{3,80}\}',
    re.IGNORECASE
)

# Any prefix{...} shaped token - cheap pre-filter for cipher brute-forcing
BRACE_CANDIDATE_REGEX = re.compile(r'[A-Za-z0-9_\-]{2,20}\{[^{}\s]{3,80}\}')


class TextDecoderHub:
    """
    Multimodal intelligence engine that tests strings against Base64, Hex, Binary,
    Morse, ROT13/Caesar, and regex to extract hidden flags and plaintext messages.
    """

    @staticmethod
    def decode_morse(morse_str):
        """Decodes morse code string with dot/dash notation."""
        cleaned = morse_str.strip().replace('_', '-').replace('/', ' / ')
        words = cleaned.split('   ')
        decoded_words = []
        for word in words:
            letters = word.strip().split(' ')
            decoded_letters = [MORSE_CODE_DICT.get(l, '') for l in letters if l]
            decoded_words.append("".join(decoded_letters))
        result = " ".join(decoded_words).strip()
        return result if result else None

    @staticmethod
    def decode_base64_token(token):
        """Decodes a single clean Base64 token."""
        try:
            # Pad if needed
            padded = token + "=" * ((4 - len(token) % 4) % 4)
            raw = base64.b64decode(padded, validate=True)
            text = raw.decode('utf-8', errors='replace')
            printable_ratio = sum(1 for c in text if 32 <= ord(c) <= 126) / max(1, len(text))
            if printable_ratio > 0.85 and len(text) >= 3:
                return text
        except Exception:
            pass
        return None

    @staticmethod
    def decode_hex_token(token):
        """Decodes a single clean Hex token."""
        if len(token) % 2 != 0:
            token = token[:-1]
        try:
            raw = bytes.fromhex(token)
            text = raw.decode('utf-8', errors='replace')
            printable_ratio = sum(1 for c in text if 32 <= ord(c) <= 126) / max(1, len(text))
            if printable_ratio > 0.85 and len(text) >= 3:
                return text
        except Exception:
            pass
        return None

    @staticmethod
    def decode_rot13(s):
        """ROT13 decoding."""
        res = []
        for c in s:
            if 'a' <= c <= 'z':
                res.append(chr((ord(c) - ord('a') + 13) % 26 + ord('a')))
            elif 'A' <= c <= 'Z':
                res.append(chr((ord(c) - ord('A') + 13) % 26 + ord('A')))
            else:
                res.append(c)
        return "".join(res)

    @staticmethod
    def _printable_text(raw, min_len=3, min_ratio=0.85):
        """Returns decoded text if the byte string is mostly printable ASCII, else None."""
        if not raw:
            return None
        text = raw.decode('utf-8', errors='replace')
        printable_ratio = sum(1 for c in text if 32 <= ord(c) <= 126 or c in "\r\n\t") / max(1, len(text))
        if printable_ratio > min_ratio and len(text.strip()) >= min_len:
            return text
        return None

    @classmethod
    def decode_binary_string(cls, bits, bitorder="big", max_bits=1 << 18):
        """
        Decodes a '0'/'1' string (whitespace/commas ignored) into text.
        bitorder='big' -> MSB first (standard), 'little' -> LSB first.
        """
        cleaned = "".join(c for c in bits[:max_bits * 2] if c in "01")[:max_bits]
        usable = (len(cleaned) // 8) * 8
        if usable < 24:
            return None
        out = bytearray()
        for i in range(0, usable, 8):
            byte = cleaned[i:i + 8]
            if bitorder == "little":
                byte = byte[::-1]
            out.append(int(byte, 2))
        return cls._printable_text(bytes(out))

    @classmethod
    def decode_base32_token(cls, token):
        """Decodes a single clean Base32 token (RFC 4648 alphabet)."""
        try:
            body = token.upper().rstrip("=")
            padded = body + "=" * ((8 - len(body) % 8) % 8)
            return cls._printable_text(base64.b32decode(padded))
        except Exception:
            return None

    @staticmethod
    def caesar_shift(s, shift):
        """Shifts alphabetic characters by `shift` positions (decoding uses a negative shift)."""
        res = []
        for c in s:
            if 'a' <= c <= 'z':
                res.append(chr((ord(c) - ord('a') + shift) % 26 + ord('a')))
            elif 'A' <= c <= 'Z':
                res.append(chr((ord(c) - ord('A') + shift) % 26 + ord('A')))
            else:
                res.append(c)
        return "".join(res)

    @staticmethod
    def decode_atbash(s):
        """Atbash cipher (A<->Z, B<->Y, ...). Self-inverse."""
        res = []
        for c in s:
            if 'a' <= c <= 'z':
                res.append(chr(ord('z') - (ord(c) - ord('a'))))
            elif 'A' <= c <= 'Z':
                res.append(chr(ord('Z') - (ord(c) - ord('A'))))
            else:
                res.append(c)
        return "".join(res)

    @classmethod
    def decode_decimal_ascii(cls, s):
        """Decodes separated decimal byte values ('102 108 97 103' -> 'flag')."""
        nums = re.findall(r'\d{2,3}', s)
        if len(nums) < 3:
            return None
        vals = [int(n) for n in nums]
        if any(v > 255 for v in vals):
            return None
        return cls._printable_text(bytes(vals), min_ratio=0.95)

    @staticmethod
    def decode_concatenated_decimal(digits):
        """
        Decodes unseparated decimal ASCII codes of mixed width ('70761231256' style).
        Codes 100-126 take three digits, 32-99 take two. Returns None if the digits
        cannot be fully consumed as printable characters.
        """
        if not digits or not digits.isdigit():
            return None
        out, i = [], 0
        while i < len(digits):
            three = digits[i:i + 3]
            if digits[i] == "1" and len(three) == 3 and 100 <= int(three) <= 126:
                out.append(chr(int(three)))
                i += 3
                continue
            two = digits[i:i + 2]
            if len(two) == 2 and 32 <= int(two) <= 99:
                out.append(chr(int(two)))
                i += 2
                continue
            return None
        return "".join(out) if len(out) >= 3 else None

    @classmethod
    def caesar_bruteforce_flags(cls, text):
        """
        Tries all Caesar shifts (except 13, reported as ROT13) on prefix{...} shaped
        candidates only, so the brute-force stays cheap on long bitstream dumps.
        Returns a list of (shift, flag) tuples.
        """
        hits = []
        for cand in set(BRACE_CANDIDATE_REGEX.findall(text)):
            for shift in range(1, 26):
                if shift == 13:
                    continue
                for f in FLAG_REGEX.findall(cls.caesar_shift(cand, -shift)):
                    hits.append((shift, f))
        return hits

    @staticmethod
    def _dedupe(results):
        """Keeps the first (highest-priority) finding for each decoded value."""
        seen = set()
        unique = []
        for r in results:
            key = r.get("decoded")
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)
        return unique

    @classmethod
    def analyze_string(cls, raw_text):
        """
        Runs comprehensive heuristics on any candidate string:
        Scans for direct flags, tokenizes base64/hex candidates, tests ROT13, and checks reversals.
        """
        results = []
        if not raw_text or len(raw_text.strip()) < 3:
            return results

        text = raw_text.strip()

        # 1. Direct Regex Match
        flags = FLAG_REGEX.findall(text)
        if flags:
            for f in flags:
                results.append({
                    "type": "Direct Flag Match",
                    "decoded": f,
                    "confidence": "VERY HIGH",
                    "is_flag": True
                })

        # 2. Tokenized Base64 Scanner (finds base64 tokens inside text)
        b64_candidates = re.findall(r'[A-Za-z0-9+/]{12,}={0,2}', text)
        for token in set(b64_candidates):
            decoded = cls.decode_base64_token(token)
            if decoded:
                flag_in_decoded = FLAG_REGEX.findall(decoded)
                if flag_in_decoded:
                    for f in flag_in_decoded:
                        results.append({
                            "type": "Base64 Decoded Flag",
                            "decoded": f,
                            "confidence": "VERY HIGH",
                            "is_flag": True
                        })
                elif len(decoded) >= 4 and any(w in decoded.lower() for w in ["flag", "secret", "key", "pass", "token"]):
                    results.append({
                        "type": "Base64 Decoded Secret",
                        "decoded": decoded,
                        "confidence": "HIGH",
                        "is_flag": True
                    })

        # 3. Tokenized Hex Scanner
        hex_candidates = re.findall(r'[0-9a-fA-F]{10,}', text)
        for token in set(hex_candidates):
            decoded = cls.decode_hex_token(token)
            if decoded:
                flag_in_decoded = FLAG_REGEX.findall(decoded)
                if flag_in_decoded:
                    for f in flag_in_decoded:
                        results.append({
                            "type": "Hex Decoded Flag",
                            "decoded": f,
                            "confidence": "VERY HIGH",
                            "is_flag": True
                        })

        # 4. ROT13 on flags
        rot_res = cls.decode_rot13(text)
        rot_flags = FLAG_REGEX.findall(rot_res)
        if rot_flags:
            for f in rot_flags:
                results.append({
                    "type": "ROT13 Decoded Flag",
                    "decoded": f,
                    "confidence": "VERY HIGH",
                    "is_flag": True
                })

        # 5. Reverse String
        rev_text = text[::-1]
        rev_flags = FLAG_REGEX.findall(rev_text)
        if rev_flags:
            for f in rev_flags:
                results.append({
                    "type": "Reversed String (Backmasking)",
                    "decoded": f,
                    "confidence": "VERY HIGH",
                    "is_flag": True
                })

        # 6. Binary strings (01100110 01101100 ...), both bit orders
        for token in re.findall(r'(?:[01]{8}[\s,]?){3,}', text):
            for order in ("big", "little"):
                decoded = cls.decode_binary_string(token, bitorder=order)
                if not decoded:
                    continue
                for f in FLAG_REGEX.findall(decoded):
                    results.append({
                        "type": f"Binary Decoded Flag ({'MSB' if order == 'big' else 'LSB'}-first)",
                        "decoded": f,
                        "confidence": "VERY HIGH",
                        "is_flag": True
                    })

        # 7. Base32 tokens
        for token in set(re.findall(r'[A-Z2-7]{8,}={0,6}', text)):
            decoded = cls.decode_base32_token(token)
            if decoded:
                for f in FLAG_REGEX.findall(decoded):
                    results.append({
                        "type": "Base32 Decoded Flag",
                        "decoded": f,
                        "confidence": "VERY HIGH",
                        "is_flag": True
                    })

        # 8. Decimal ASCII sequences (102 108 97 103 ...)
        for token in re.findall(r'(?:\b\d{2,3}\b[\s,;]+){3,}\b\d{2,3}\b', text):
            decoded = cls.decode_decimal_ascii(token)
            if decoded:
                for f in FLAG_REGEX.findall(decoded):
                    results.append({
                        "type": "Decimal ASCII Decoded Flag",
                        "decoded": f,
                        "confidence": "VERY HIGH",
                        "is_flag": True
                    })

        # 9. Caesar brute-force and Atbash on prefix{...} candidates
        if BRACE_CANDIDATE_REGEX.search(text):
            for shift, f in cls.caesar_bruteforce_flags(text):
                results.append({
                    "type": f"Caesar Decoded Flag (shift={shift})",
                    "decoded": f,
                    "confidence": "HIGH",
                    "is_flag": True
                })
            for cand in set(BRACE_CANDIDATE_REGEX.findall(text)):
                for f in FLAG_REGEX.findall(cls.decode_atbash(cand)):
                    results.append({
                        "type": "Atbash Decoded Flag",
                        "decoded": f,
                        "confidence": "HIGH",
                        "is_flag": True
                    })

        return cls._dedupe(results)
