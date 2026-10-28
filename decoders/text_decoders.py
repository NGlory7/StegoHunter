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

        return results
