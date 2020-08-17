import os
import re
import unidecode


class DataManager(object):
    NAMES_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "names")

    def __init__(self):
        all_files = [
            os.path.join(DataManager.NAMES_PATH, filename)
            for filename in os.listdir(DataManager.NAMES_PATH)
        ]

        self.data_dict = {}
        for filepath in all_files:
            filename = os.path.basename(filepath)
            base_name = ".".join(filename.split(".")[:-1]).lower()
            self.data_dict[base_name] = filepath

    def get_origins(self):
        return list(self.data_dict.keys())

    def get_names(self, origin):
        filepath = self.data_dict[origin]
        with open(filepath, "r", encoding="utf-8") as f:
            content = re.sub(
                r"[^a-z \n\-\']+", "", unidecode.unidecode(f.read().lower())
            )
            return [name for name in content.split("\n") if len(name) > 0]

    def get_charlist(self):
        charset = set()
        for origin in self.get_origins():
            names = self.get_names(origin)
            for n in names:
                for c in n:
                    charset.add(c)
        return list(charset)


if __name__ == "__main__":
    dm = DataManager()
    origins = dm.get_origins()
    print(sorted(dm.get_charlist()))
