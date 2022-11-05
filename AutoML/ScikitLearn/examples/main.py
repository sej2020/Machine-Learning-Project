import sys
import os
sys.path.append(os.getcwd())

from examples.utils import append_word

msg = append_word("Hello and ")
print(msg)