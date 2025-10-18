from os import getenv

PANDAS: bool = getenv("PANDAS", "0") == "1"
