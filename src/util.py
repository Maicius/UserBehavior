import numpy as np

extract_age_pattern = '(?:\[([0-9]{1,2})\,([0-9]{1,2})\])|(?:\>=(60))'
def complete_binary(num, length):
    if len(num) < length:
        num = '0' * (length - len(num)) + num
    return num