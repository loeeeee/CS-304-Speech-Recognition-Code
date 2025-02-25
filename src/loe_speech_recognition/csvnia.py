# Auftakt
from typing import List, Union, Dict
import logging

logger = logging.getLogger(__name__)

# Why do I write this? Why? Why? Why? Why?

class CSV:
    def __init__(self, columns: List[str]) -> None:
        self.columns: List[str] = columns
        self.records: List[List] = []
        logger.debug(f"Finish initialize the CSV.")

    def __str__(self) -> str:
        columns = ", ".join(self.columns)
        size = len(self)
        return f"Columns: {columns} Size: {size}"

    def __len__(self):
        return len(self.records)

class CSVWriter(CSV):
    def __init__(self, columns: List[str]) -> None:
        super().__init__(columns)

    def add_line(self, line: List) -> None:
        self.records.append(line)
        return

    def write(self, path: str) -> None:
        records_str: str = f"{self.line_escape(self.columns)}\n"
        for line in self.records:
            escaped_line = self.line_escape(line)
            records_str += f"{escaped_line}\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(records_str)
        logger.info(f"Finish writing CSV to {path}")
        return

    @staticmethod
    def line_escape(line: List) -> str:
        result: List = []
        for entry in line:
            if isinstance(entry, str):
                escaped_value = str(entry).replace('"', '""')
                quoted_value = f"\"{escaped_value}\""
                result.append(quoted_value)
            else:
                result.append(str(entry))
        return "|".join(result)


class CSVReader(CSV):
    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for index, line in enumerate(f.readlines()):
                line = line.strip()
                if index == 0:
                    super().__init__([i.replace("\"", "") for i in line.split("|")])
                else:
                    parsed_line = self.line_parser(line)
                    self.records.append(parsed_line)
        logger.info(f"Read CSV from {path}")
        self._index: int = -1

    def __next__(self) -> Dict[str, Union[str, None, int]]:
        self._index += 1
        if self._index == len(self):
            raise StopIteration
        result: Dict = {}
        for key, value in zip(self.columns, self.records[self._index]):
            result[key] = value
        return result

    def __iter__(self):
        return self

    @staticmethod
    def line_parser(line: str) -> List[Union[str, int, None]]:
        parsed_line: List[str] = line.split("|")
        result: List[Union[str, int, None]] = []
        for entry in parsed_line:
            if entry[0] == "\"" and entry[-1] == "\"":
                entry = entry[1:-1]
                entry = entry.replace("\"\"", "\"")
            elif entry == "None":
                entry = None
            elif entry.isdigit():
                entry = int(entry)
            
            result.append(entry)
        return result