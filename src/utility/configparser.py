# -*- coding: utf-8 -*-
"""

"""
import string
from debug import NNDebug


class ConfigParser:

    def __init__(self):
        pass

    @staticmethod
    def error(msg):
        NNDebug.error("[Configuration] " + msg)

    @staticmethod
    def remove_quote(line):
        return line[:line.find("#")].strip()

    def parse(self, config_path, expand=False):
        try:
            config_file = open(config_path)
            lines = map(self.remove_quote, config_file.readlines())
            config_file.close()
            if expand:
                content_array = self.expand_content(lines)
                configs = []
                for lines in content_array:
                    config = self.read_content(lines)
                    configs.append(config)
                return configs
            else:
                config = self.read_content(lines)
                return config
        except IOError:
            self.error("unable to open configuration file '%s'" % config_path)
            return None

    @staticmethod
    def expand_content(lines):
        results = []
        accu = []

        def dfs(idx):
            if idx == len(lines):
                result = [line for line in accu]
                results.append(result)
                return
            else:
                line = lines[idx]
                segs = line.split("=")
                if len(segs) == 2:
                    value = segs[1].strip()
                    if value.startswith("{") and value.endswith("}"):
                        terms = value[1:-1].split(",")
                        for term in terms:
                            selected_line = segs[0] + " = " + term
                            accu.append(selected_line)
                            dfs(idx+1)
                            accu.pop()
                        return
                accu.append(line)
                dfs(idx+1)
        dfs(0)
        return results

    def read_content(self, lines):
        i = 0
        list_sections = {"paramlist": "param"}
        singular_sections = ["training"]
        d = {
            "param": [],
            "input": [],
            "weight": [],
            "group": [],
            "layer": [],
            "loss": [],
            "training": None
        }
        while i < len(lines):
            name, block, new_i = self.read_block(lines, i)
            if not block:
                break
            if name in list_sections:
                d[list_sections[name]] += [{"name": k, "value": v} for k, v in block.items()]
            elif name in singular_sections:
                d[name] = block
            elif name in d:
                d[name].append(block)
            else:
                self.error("invalid block name '%s' at line %d" % (name, i))
            i = new_i
        return d

    def read_block(self, lines, i):
        while i < len(lines):
            line = lines[i]
            if line.startswith("[") and line.endswith("]"):
                break
            i += 1
        else:
            return "", None, i
        block = {}
        name = lines[i][1:-1].strip()
        i += 1
        while i < len(lines):
            line = lines[i].strip()
            if line == "":
                i += 1
                continue
            elif line.startswith("["):
                break
            else:
                eq_idx = line.find("=")
                if eq_idx < 0:
                    self.error("invalid block content at line %d" % i)
                key = line[:eq_idx].strip()
                value = line[eq_idx+1:].strip()
                block[key] = self.eval(value, i)
                i += 1
        return name, block, i

    def eval(self, v, linenu):
        v = v.strip()
        error_msg = "invalid value '%s' at line %d" % (v, linenu)
        # '[...]' formatted array
        if v.startswith("["):
            if v.endswith("]"):
                return [self.eval(_, linenu) for _ in v[1:-1].split(",")]
            else:
                self.error(error_msg)
        # pure comma splitted array
        elif v.find(",") >= 0:
            return [self.eval(_, linenu) for _ in v.split(",")]
        # string parameter "..."
        elif v.startswith('"'):
            if v.endswith('"'):
                return v
            else:
                self.error(error_msg)
        # string parameter '...'
        elif v.startswith("'"):
            if v.endswith("'"):
                return v
            else:
                self.error(error_msg)
        # arithmetic format
        else:
            if v.find("+") > 0:
                v = sum([self.eval(_, linenu) for _ in v.split("+")])
            else:
                try:
                    v = string.atoi(v)
                except ValueError:
                    try:
                        v = string.atof(v)
                    except ValueError:
                        self.error(error_msg)
                finally:
                    return v
