# -*- coding: utf-8 -*-
"""

"""
import string

from debug import NNDebug


class ConfigParser:

    def __init__(self):
        pass

    @staticmethod
    def remove_quote(line):
        return line[:line.find("#")].strip()

    def parse(self, config_path):
        try:
            config_file = open(config_path)
        except IOError:
            NNDebug.error("unable to open configuration file '%s" % config_path)
            return None

        lines = map(self.remove_quote, config_file.readlines())
        config_file.close()

        idx = 0
        length = len(lines)

        list_sections = {"paramlist": "param"}
        single_sections = ["training"]

        d = {
            "param": [],
            "input": [],
            "weight": [],
            "layer": [],
            "loss": [],
            "training": None
        }

        def eval_(v):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                return map(eval_,
                           v[1:-1].split(","))
            elif v.find(",") >= 0:
                return map(eval_,
                           filter(lambda _: _.strip() != "",
                                  v.split(",")))
            elif v.startswith("{"):
                return dict([
                    (key, eval_(value))
                    for key, value in
                    filter(lambda _: len(_) == 2,
                           [_.split(":") for _ in v[1:-1].split(",")])
                ])
            else:
                try:
                    v = string.atoi(v)
                except ValueError:
                    try:
                        v = string.atof(v)
                    except ValueError:
                        pass
                finally:
                    return v

        def read_block_(i):
            while i < length:
                line = lines[i]
                if line.startswith("[") and \
                        line.endswith("]"):
                    break
                i += 1
            else:
                return i

            item = {}
            section = lines[i][1:-1].strip()
            i += 1

            while i < length:
                line = lines[i]
                if line.startswith("["):
                    break
                i += 1
                eq_idx = line.find("=")
                if eq_idx < 0:
                    continue
                key = line[:eq_idx].strip()
                value = line[eq_idx+1:].strip()
                item[key] = eval_(value)

            if section in list_sections:
                d[list_sections[section]] += \
                    [{"name": k, "value": v}
                     for k, v in item.items()]
            elif section in single_sections:
                d[section] = item
            elif section in d:
                d[section].append(item)

            return i

        while idx < length:
            idx = read_block_(idx)
        return d
