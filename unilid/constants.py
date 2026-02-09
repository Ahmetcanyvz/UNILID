import random
import math
import re
from collections import OrderedDict


SPECIAL_TOKENS = OrderedDict([("bos_token", "<s>"), ("eos_token", "</s>"), ("pad_token","<pad>"), ("unk_token","<unk>")])
NUM_TRIMMING_ITERATIONS = 3
INITIAL_VOCAB_MULT_FACTOR = 8
MIN_CHAR_OCCURENCE_THRESHOLD = 2
MIN_TOKEN_PROB = 1e-12
MIN_TOKEN_LOG_PROB = math.log(MIN_TOKEN_PROB)
MAX_BYTE_TOKEN_LEN = 50
MAX_CHAR_TOKEN_LEN = 35
MIN_TOKENIZER_LANG_TRAINING_LINES = 5000
TOKENIZER_TOTAL_TRAINING_LINES = 2_000_000
MAX_VAL_TEST_SIZE = 3000
_RNG = random.Random(42)
_SP_BYTE_ANY_RE = re.compile(r"<0x([0-9A-F]{2})>")
_SP_SPACE = "▁"


def _build_gpt2_byte_maps():
    """
    Standard GPT-2 byte<->unicode mapping used by HF ByteLevel.
    Returns:
        byte_to_unicode: dict[int -> str(one-char)]
        unicode_to_byte: dict[str(one-char) -> int]
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_to_unicode = dict(zip(bs, [chr(c) for c in cs]))
    unicode_to_byte = {u: b for b, u in byte_to_unicode.items()}
    return byte_to_unicode, unicode_to_byte


_BYTE_TO_UNI, _UNI_TO_BYTE = _build_gpt2_byte_maps()
_HF_SPACE = _BYTE_TO_UNI[32]  # typically "Ġ"

SP_DEFAULT_ARGS = ["--model_type=unigram",
                    "--normalization_rule_name=identity",
                    "--remove_extra_whitespaces=false",
                    "--allow_whitespace_only_pieces=true",
                    "--unk_id=0",
                    "--add_dummy_prefix=false",
                    "--split_by_whitespace=false",  # handled by write_hf_bytelevel_corpus
                    "--split_by_unicode_script=false",
                    "--split_by_number=false"]
