
import torch
import torch.nn as nn
import struct
import numpy as np
from typing import Tuple




def is_fp32(t): return t.dtype == torch.float32
def is_fp16(t): return t.dtype == torch.float16
def is_bf16(t): return t.dtype == torch.bfloat16


# fp16 value  →  16-bit binary string
def fp16_to_bits(value) -> str:
    """
    Return the exact 16-bit IEEE-754 representation of `value`
    as a zero-padded binary string.
    Accepts  • numpy.float16   • Python float   • int/str bit pattern
    """
    val16  = np.float16(value)                # round to fp16 if needed
    uint16 = val16.view(np.uint16).item()     # reinterpret bits
    return f"{uint16:016b}"                   # 16 chars



# 16-bit binary string  →  fp16 value
def bits_to_fp16(bits):
    """
    `bits` may be
       • 16-char bit string  ('0100…')
       • int in range 0-65535
    Returns: numpy.float16  (exact same payload)
    """
    if isinstance(bits, str):
        bits = int(bits, 2)
    return np.uint16(bits).view(np.float16)



# fp16 value  →  16-bit binary string
def bf16_to_bits(value) -> str:
    fp32_bits = np.float32(value).view(np.uint32).item()
    return f"{fp32_bits >> 16:016b}"          # keep upper 16 bits



# 16-bit binary string  →  fp16 value
def bits_to_bf16(bits):
    if isinstance(bits, str):
        bits = int(bits, 2)
    fp32_bits = bits << 16
    return np.uint32(fp32_bits).view(np.float32)  # result is numpy.float32



def fp32_to_bits(value):        # """Convert a 32-bit floating point number to IEEE 754 binary string format."""
    # Pack float into 4 bytes using IEEE 754 standard
    packed = struct.pack('!f', value)  # '!f' means big-endian float (32-bit)
    
    # Convert packed bytes to an unsigned 32-bit integer
    int_rep = struct.unpack('!I', packed)[0]    
    # else output in string format
    binary_rep = f"{int_rep:032b}"      # Convert to 32-bit binary string
    return binary_rep



def bits_to_fp32(bits):

    if isinstance(bits, str):
        if len(bits) != 32 or set(bits) - {"0", "1"}:
            raise ValueError("Expect a 32-bit binary string")
        int_rep = int(bits, 2)                      # str → int

    # Pack the 32-bit integer into 4 bytes, then unpack as float
    packed = struct.pack('!I', int_rep)             # big-endian uint32
    value  = struct.unpack('!f', packed)[0]      # big-endian float32
    return value




def flip_bit_in_bitstring_at_position_n(bit_str: str, n: int, flip_direction="all"):           # flip_dir = "01"/"10"/"all"=> all means flip no matter what direction
    """
    Bit indexing (MSB → LSB):
       n = 0  → sign bit (bit-15)
       n = 1  → MSB of exponent 
       n = 2-7→ remaining exponent bits (LSB of exponent is n = 7)
       n = 8-31 -> mantissa bits
    """
    if set(bit_str) - {"0", "1"}:
        raise ValueError("bit_str_16 must be a binary string")

    if not (0 <= n < 32) or n>=len(bit_str) :
        raise ValueError("n must be in the range 0-31 and less than string length")

    # flip n_th position
    success = 0
    bit_val = bit_str[n]
    if flip_direction == "all":
        flipped_bit = "0" if bit_val == "1" else "1"
        success = 1

    elif flip_direction == "01":
        flipped_bit = "1" 
        if bit_val == "0":
            success = 1 

    elif flip_direction == "10":
        flipped_bit = "0"
        if bit_val == "1":
            success = 0 

    else:
        print("error : Direction not recognized -> not flipping anything")
        return 0, bit_str
    
    modified_str = bit_str[:n] + flipped_bit + bit_str[n + 1:]
    return success, modified_str



def flip_bit_in_fp32(elem, n, flip_direction="all"):                # flip_dir = "01"/"10"/"all"=> all means flip no matter what direction
    bit_str = fp32_to_bits(elem)
    success, bit_str_flipped = flip_bit_in_bitstring_at_position_n(bit_str, n, flip_direction)

    return success, bits_to_fp32(bit_str_flipped)

    # not optimizing right now. So, let the code go to str->fp32 function
    # if flip_direction == "all":
    #     return 1, bits_to_fp32(bit_str_flipped)
    # if success:
    #     return success, bits_to_fp32(bit_str_flipped)
    # else:
    #     return success, elem        # need not change. direction already satisfied



def flip_bit_in_fp16(elem, n, flip_direction="all"):                # flip_dir = "01"/"10"/"all"=> all means flip no matter what direction
    bit_str = fp16_to_bits(elem)
    success, bit_str_flipped = flip_bit_in_bitstring_at_position_n(bit_str, n, flip_direction)
    return success, bits_to_fp16(bit_str_flipped)



def flip_bit_in_bf16(elem, n, flip_direction="all"):                # flip_dir = "01"/"10"/"all"=> all means flip no matter what direction
    bit_str = bf16_to_bits(elem)
    success, bit_str_flipped = flip_bit_in_bitstring_at_position_n(bit_str, n, flip_direction)
    return success, bits_to_bf16(bit_str_flipped)



def flip_bits_at_position(element, n: int, flip_direction: str = "all") -> Tuple[int, str]:                # flip_direction = "01"/"10"/"all"=> all means flip no matter what direction
    if is_fp32(element):
        return flip_bit_in_fp32(element, n, flip_direction)
    
    elif is_fp16(element):
        return flip_bit_in_fp16(element, n, flip_direction)
    
    elif is_bf16(element):
        return flip_bit_in_bf16(element, n, flip_direction)
    
    else:
        raise ValueError("element not in FP32, FP16, or BF16 floating point type")



